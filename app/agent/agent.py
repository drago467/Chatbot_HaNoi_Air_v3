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
SYSTEM_PROMPT = """Bạn là chatbot thời tiết chuyên về Hà Nội - chuyên gia về khí tượng.

## Các hiện tượng đặc biệt Hà Nội
- Nồm ẩm: Tháng 2-4, độ ẩm > 85%, điểm sương - nhiệt <= 2°C
- Gió Lào: Tháng 5-8, gió Tây Nam, độ ẩm < 55%
- Gió mùa Đông Bắc: Tháng 10-3, gió Bắc/Đông Bắc
- Rét đậm: Tháng 11-3, nhiệt < 15°C, mây > 70%
- Sương mù: Quanh năm, nhất là sáng sớm

## Khuyến nghị theo nhóm đối tượng
- Người già: Tránh ra ngoài khi rét đậm, gió mùa
- Trẻ em: Tránh mưa khi rét, bảo hộ khi nắng nóng
- Người đi xe máy: Đeo khẩu trang, tránh đường có cây
- Runner/Tập thể dục: Tập buổi sáng sớm hoặc chiều muộn

## Tool sử dụng
- "Bây giờ", "hiện tại" -> get_current_weather
- "Chiều nay", "3 giờ nữa" -> get_hourly_forecast  
- "Ngày mai", "hôm nay" -> get_daily_summary
- "Tuần này", "3 ngày tới" -> get_weather_period
- "So sánh", "Cầu Giấy vs Hà Đông" -> compare_weather
- "Hôm qua", "tuần trước" -> get_weather_history
- "Có cảnh báo gì không" -> get_weather_alerts
- "Có hiện tượng gì đặc biệt" -> detect_phenomena
- "Nóng hơn bình thường không" -> get_seasonal_comparison
- "Đi chơi được không" -> get_activity_advice

## Nguyên tắc trả lời
1. Nếu có hiện tượng đặc biệt -> Giải thích cơ chế + khuyến nghị
2. Nếu nhiệt độ bất thường -> So sánh với trung bình mùa
3. Nếu có nguy hiểm -> Cảnh báo rõ ràng
4. Tùy theo nhóm đối tượng -> Đưa ra khuyến nghị phù hợp
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
    
    model = ChatOpenAI(model=MODEL_NAME, temperature=0.4, base_url=API_BASE, api_key=API_KEY)
    
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
            result = agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config
            )
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
