"""LangGraph Agent for Weather Chatbot."""

import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI

from app.agent.tools import TOOLS


# System prompt
SYSTEM_PROMPT = """Ban la chatbot thoi tiet chuyen ve Ha Noi - chuyen gia ve khi tuong.

## Cac hien tuong dac biet Ha Noi
- Nom am: Thang 2-4, do am > 85%, dew_point - temp <= 2C
- Gio Lao: Thang 5-8, gio Tay Nam, do am < 55%
- Gio mua Dong Bac: Thang 10-3, gio Bac/Dong Bac
- Ret dam: Thang 11-3, nhiet < 15C, may > 70%
- Suong mu: Quanh nam, nhat la sang som

## Khuyen nghi theo nhom doi tuong
- Nguoi gia: Tranh ra ngoai khi ret dam, gio mua
- Tre em: Tranh mua khi ret, bao ho khi nang nong
- Nguoi di xe may: Deo khau trang, tranh duong co cay
- Runner/Tap the duc: Tap buoi sang som hoac chieu muon

## Tool su dung
- "Bay gio", "hien tai" -> get_current_weather
- "Chieu nay", "3 gio nua" -> get_hourly_forecast  
- "Ngay mai", "hom nay" -> get_daily_summary
- "Tuan nay", "3 ngay toi" -> get_weather_period
- "So sanh", "Cau Giay vs Ha Dong" -> compare_weather
- "Hom qua", "tuan truoc" -> get_weather_history
- "Co canh bao gi khong" -> get_weather_alerts
- "Co hien tuong gi dac biet" -> detect_phenomena
- "Nong hon binh thuong khong" -> get_seasonal_comparison
- "Di choi duoc khong" -> get_activity_advice

## Nguyen tac tra loi
1. Neu co hien tuong dac biet -> Giai thich co che + khuyen nghi
2. Neu nhiet do bat thuong -> So sanh voi trung binh mua
3. Neu co nguy hiem -> Canh bao ro rang
4. Tuy theo nhom doi tuong -> Dua ra khuyen nghi phu hop
"""

# Cache agent instance
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_weather_agent()
    return _agent

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
    
    checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
    checkpointer.setup()
    
    agent = create_react_agent(model=model, tools=TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
    return agent

def run_agent(message: str, thread_id: str = "default") -> dict:
    """Run agent synchronously (blocking).
    
    Also logs tool calls to evaluation_logger.
    """
    import time
    
    # Get logger
    try:
        from app.agent.evaluation_logger import get_evaluation_logger
        logger = get_evaluation_logger()
    except:
        logger = None
    
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Wrap tools to log calls (if logger available)
    if logger:
        # We'll log after getting results
        pass
    
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
    
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
    
    Args:
        message: User message
        thread_id: Conversation thread ID
        
    Yields:
        Text chunks from the agent's response
    """
    from langchain.schema.messages import AIMessageChunk, ToolMessage
    
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
    from langchain.schema.messages import ToolMessage
    import time
    
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get logger
    try:
        from app.agent.evaluation_logger import get_evaluation_logger
        logger = get_evaluation_logger()
    except:
        logger = None
    
    tool_start_time = None
    current_tool_name = None
    
    # Stream with both messages and updates
    for event in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode=["messages", "updates"]
    ):
        # Handle different event formats
        if isinstance(event, tuple):
            msg_chunk, metadata = event
            
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
                yield {"type": "tool", "content": msg_chunk}
        elif isinstance(event, dict):
            yield event

if __name__ == "__main__":
    print("Agent module ready!")
