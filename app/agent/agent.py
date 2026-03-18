"""LangGraph Agent for Weather Chatbot."""

import os
import threading
from dotenv import load_dotenv

load_dotenv()

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
import psycopg

from app.agent.tools import TOOLS


# System prompt
SYSTEM_PROMPT = """Ban la tro ly thoi tiet chuyen ve Ha Noi. CHI tra loi ve thoi tiet khu vuc Ha Noi.
Phong cach: than thien, chuyen nghiep, ngan gon, dung tieng Viet tu nhien.

## Quy tac chon tool
- "bay gio", "hien tai", "dang" -> get_current_weather (phuong) hoac get_district_weather / get_city_weather
- "chieu nay", "toi nay", "3 gio nua", "sang mai" -> get_hourly_forecast
- "ngay mai", "hom nay" (ca ngay) -> get_daily_summary
- "tuan nay", "3 ngay toi", "cuoi tuan" -> get_weather_period
- "hom qua", "tuan truoc" -> get_weather_history
- "quan nao nong nhat", "top", "xep hang" -> get_district_ranking
- "phuong nao trong quan X" -> get_ward_ranking_in_district
- "mua den bao gio", "may gio tanh", "khi nao mua" -> get_rain_timeline
- "may gio tot nhat", "luc nao nen" -> get_best_time
- "mac gi", "can ao khoac khong", "mang o khong" -> get_clothing_advice
- "am len khi nao", "xu huong nhiet", "bao gio het ret" -> get_temperature_trend
- "so sanh A va B" -> compare_weather
- "co canh bao gi" -> get_weather_alerts
- "co hien tuong gi" -> detect_phenomena
- "nong hon binh thuong khong" -> get_seasonal_comparison
- "di choi duoc khong", "chay bo duoc khong" -> get_activity_advice

## Quy uoc thoi gian (ICT = UTC+7)
- "sang" = 6h-11h, "trua" = 11h-13h, "chieu" = 13h-18h, "toi" = 18h-22h, "dem" = 22h-6h
- "cuoi tuan" = Thu 7 + Chu nhat tuan nay (hoac tuan toi neu da qua)
- "tuan nay" = tu hom nay den Chu nhat

## Dia diem noi tieng (POI)
- Ho Guom, Ho Hoan Kiem -> quan Hoan Kiem
- My Dinh, San My Dinh -> quan Nam Tu Liem
- Ho Tay -> quan Tay Ho
- San bay Noi Bai -> huyen Soc Son
- Times City -> quan Hai Ba Trung
- Cong vien Cau Giay -> quan Cau Giay
- Van Mieu -> quan Dong Da
- Lang Bac -> quan Ba Dinh

## Luu y ve du lieu
- Du lieu HIEN TAI khong co xac suat mua (pop) -> khi hoi "co mua khong?",
  check weather_main (Rain/Drizzle/Thunderstorm) + goi them get_hourly_forecast 1-2h toi
- rain_1h chi co khi dang mua -> NULL khong co nghia la khong mua
- Du lieu LICH SU thieu visibility va UV -> khong hua tra cac thong so nay cho qua khu
- wind_gust co the NULL khi gio nhe -> dung wind_speed thay the

## Cac hien tuong dac biet Ha Noi
- Nom am: Thang 2-4, do am > 85%, diem suong - nhiet <= 2C
- Gio Lao: Thang 5-8, gio Tay Nam, do am < 55%
- Gio mua Dong Bac: Thang 10-3, gio Bac/Dong Bac
- Ret dam: Thang 11-3, nhiet < 15C, may > 70%
- Suong mu: Quanh nam, nhat la sang som

## Dinh dang tra loi
- Cho quan/thanh pho: tong quan + top phuong nong/lanh nhat + hien tuong dac biet
- Cho phuong: chi tiet day du cac thong so
- Luon kem khuyen nghi thuc te khi co hien tuong dac biet
- Khi co nhieu thong tin, dung bullet points de de doc

## Khi can goi nhieu tool
- "Thoi tiet Ha Noi hom nay" -> get_city_weather + get_district_ranking(nhiet_do)
- "Co nen di choi khong" -> get_best_time + get_clothing_advice
- "Quan Cau Giay thoi tiet the nao" -> get_district_weather + get_ward_ranking_in_district

## Xu ly loi
- Khong tim thay dia diem -> Goi y: "Ban co the noi ro hon? Vi du: quan Cau Giay"
- Khong co du lieu -> "Hien chua co du lieu cho [X]. Thu [Y] nhe?"
- Du lieu cu -> Canh bao ro rang thoi gian cap nhat
"""

# Thread-safe agent cache
_agent = None
_agent_lock = threading.Lock()
_db_connection = None


def get_agent():
    """Get or create the weather agent (thread-safe)."""
    global _agent
    if _agent is None:
        with _agent_lock:
            # Double-check after acquiring lock
            if _agent is None:
                _agent = create_weather_agent()
    return _agent


def reset_agent():
    """Reset the cached agent to force recreation with fresh connections."""
    global _agent
    global _db_connection
    with _agent_lock:
        # Close the database connection before resetting
        if _db_connection is not None:
            try:
                _db_connection.close()
            except:
                pass
            _db_connection = None
        _agent = None


def create_weather_agent():
    API_BASE = os.getenv("API_BASE")
    API_KEY = os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL", "gpt-4o-2024-11-20")
    
    if not API_BASE or not API_KEY:
        raise ValueError("API_BASE and API_KEY must be set in .env")
    
    model = ChatOpenAI(model=MODEL_NAME, temperature=0, base_url=API_BASE, api_key=API_KEY)
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")
    
    # Create connection and keep it alive as part of checkpointer
    # The checkpointer will use this connection for checkpointing
    import psycopg
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    
    # Store connection in global so it doesn't get garbage collected
    global _db_connection
    _db_connection = conn
    
    agent = create_react_agent(model=model, tools=TOOLS, state_modifier=SYSTEM_PROMPT, checkpointer=checkpointer)
    
    return agent

def run_agent(message: str, thread_id: str = "default") -> dict:
    """Run agent synchronously (blocking).
    
    Also logs tool calls to evaluation_logger.
    Includes automatic retry on connection errors.
    """
        
    # Get logger
    try:
        from app.agent.evaluation_logger import get_evaluation_logger
        logger = get_evaluation_logger()
    except Exception:
        logger = None
    
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Wrap tools to log calls (if logger available)
    if logger:
        # We'll log after getting results
        pass
    
    # Retry logic for stale connections
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
                agent = get_agent()
            else:
                raise last_error
    
    # Extract and log tool calls from result
    if logger:
        try:
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.log_tool_call(
                            session_id=thread_id,
                            turn_number=0,
                            tool_name=tc.get("name", "unknown"),
                            tool_input=str(tc.get("args", {}))[:200],
                            tool_output="",
                            success=True,
                            execution_time_ms=0
                        )
        except Exception as e:
            pass  # Don't break on logging errors
    
    return result


def stream_agent(message: str, thread_id: str = "default"):
    """Stream agent response token by token.
    
    Yields chunks of the response for real-time display.
    Only yields LLM text (AIMessageChunk from node "agent").
    
    Includes automatic retry on connection errors.
    
    Args:
        message: User message
        thread_id: Conversation thread ID
        
    Yields:
        Text chunks from the agent's response
    """
    from langchain_core.messages import ToolMessage, AIMessageChunk
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream with "messages" mode to get token-by-token updates
            for event in agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config,
                stream_mode="messages"
            ):
                # event is a tuple of (message_chunk, metadata)
                if event and len(event) >= 2:
                    msg_chunk, metadata = event
                    
                    # Skip tool messages (they contain raw JSON from DAL)
                    if isinstance(msg_chunk, ToolMessage):
                        continue
                    
                    # Skip messages with tool_calls (function calling JSON)
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue
                    
                    # Only yield content from agent node, not tools node
                    if metadata.get("langgraph_node") == "agent":
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            yield msg_chunk.content
            return  # Success, exit function
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
            else:
                raise last_error


def stream_agent_with_updates(message: str, thread_id: str = "default"):
    """Stream agent response with both messages and tool updates.
    
    Yields dict with 'type' and 'content' keys:
    - type='message': text chunk from LLM
    - type='tool': tool call start/update/end
    
    Also logs tool calls to evaluation_logger.
    
    Args:
        message: User message
        thread_id: Conversation thread ID
        
    Yields:
        Dict with type and content
    """
    from langchain_core.messages import ToolMessage, AIMessageChunk
        
    # Retry logic for stale connections
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get logger
            try:
                from app.agent.evaluation_logger import get_evaluation_logger
                logger = get_evaluation_logger()
            except Exception:
                logger = None
            
            # Stream with both messages and updates
            for event in agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config,
                stream_mode=["messages", "updates"]
            ):
                # Handle different event formats from LangGraph
                # When stream_mode is a list, events come as (stream_name, event_data)
                if isinstance(event, tuple) and len(event) == 2:
                    stream_name, event_data = event
                    
                    if stream_name == "messages":
                        # event_data is (chunk, metadata)
                        if isinstance(event_data, tuple) and len(event_data) == 2:
                            msg_chunk, metadata = event_data
                            
                            # Skip tool messages (raw JSON from DAL)
                            if isinstance(msg_chunk, ToolMessage):
                                continue
                            
                            # Skip messages with tool_calls (function calling JSON)
                            if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                                continue
                            
                            # Message chunk from agent node
                            if metadata.get("langgraph_node") == "agent":
                                if hasattr(msg_chunk, "content") and msg_chunk.content:
                                    yield {"type": "message", "content": msg_chunk.content}
                            
                            # Tool updates (from tools node)
                            if metadata.get("langgraph_node") == "tools":
                                yield {"type": "tool", "content": msg_chunk if isinstance(msg_chunk, str) else str(msg_chunk)}
                    
                    elif stream_name == "updates":
                        # event_data is dict with tool outputs
                        yield {"type": "tool", "content": event_data}
            
            return  # Success, exit function
        
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Reset agent to get fresh connection
                reset_agent()
            else:
                raise last_error
