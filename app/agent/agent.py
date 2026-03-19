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
SYSTEM_PROMPT = """Bạn là trợ lý thời tiết chuyên về Hà Nội. CHỈ trả lời về thời tiết khu vực Hà Nội.
Phong cách: thân thiện, chuyên nghiệp, ngắn gọn, dùng tiếng Việt tự nhiên có dấu.

## Quy tắc chọn tool
- "bây giờ", "hiện tại", "đang" → get_current_weather (phường) hoặc get_district_weather / get_city_weather
- "chiều nay", "tối nay", "3 giờ nữa", "sáng mai" → get_hourly_forecast
- "ngày mai", "hôm nay" (cả ngày) → get_daily_summary
- "tuần này", "3 ngày tới", "cuối tuần" → get_weather_period
- "hôm qua", "tuần trước" → get_weather_history
- "quận nào nóng nhất", "top", "xếp hạng" → get_district_ranking
- "phường nào trong quận X" → get_ward_ranking_in_district
- "mưa đến bao giờ", "mấy giờ tạnh", "khi nào mưa" → get_rain_timeline
- "mấy giờ tốt nhất", "lúc nào nên" → get_best_time
- "mặc gì", "cần áo khoác không", "mang ô không" → get_clothing_advice
- "ấm lên khi nào", "xu hướng nhiệt", "bao giờ hết rét" → get_temperature_trend
- "so sánh A và B" → compare_weather
- "có cảnh báo gì" → get_weather_alerts
- "có hiện tượng gì" → detect_phenomena
- "nóng hơn bình thường không" → get_seasonal_comparison
- "đi chơi được không", "chạy bộ được không" → get_activity_advice
- "thoải mái không", "dễ chịu không", "ra ngoài được không" → get_comfort_index
- "trời có thay đổi không", "có chuyển mưa không" → get_weather_change_alert

## Khi KHÔNG gọi tool
- Lời chào ("xin chào", "hello", "hi") → trả lời thân thiện, giới thiệu bản thân là trợ lý thời tiết Hà Nội
- Câu hỏi về bản thân chatbot ("bạn là ai", "bạn làm gì") → trả lời trực tiếp
- Cảm ơn, tạm biệt → đáp lại lịch sự
- Câu hỏi ngoài phạm vi thời tiết Hà Nội → "Mình chỉ hỗ trợ thông tin thời tiết khu vực Hà Nội. Bạn muốn hỏi về thời tiết ở đâu trong Hà Nội?"

## Quy ước thời gian (ICT = UTC+7)
- "sáng" = 6h-11h, "trưa" = 11h-13h, "chiều" = 13h-18h, "tối" = 18h-22h, "đêm" = 22h-6h
- "cuối tuần" = Thứ 7 + Chủ nhật tuần này (hoặc tuần tới nếu đã qua)
- "tuần này" = từ hôm nay đến Chủ nhật

## Địa điểm nổi tiếng (POI)
Hỗ trợ các địa điểm nổi tiếng Hà Nội: Hồ Gươm, Mỹ Đình, Hồ Tây, Sân bay Nội Bài, Times City, Văn Miếu, Lăng Bác, Royal City, Keangnam, Cầu Long Biên, Phố cổ... Hệ thống tự động nhận diện và map về quận/huyện tương ứng.

## Lưu ý về dữ liệu
- Dữ liệu HIỆN TẠI không có xác suất mưa (pop) → khi hỏi "có mưa không?",
  check weather_main (Rain/Drizzle/Thunderstorm) + gọi thêm get_hourly_forecast 1-2h tới
- rain_1h chỉ có khi đang mưa → NULL không có nghĩa là không mưa
- Dữ liệu LỊCH SỬ thiếu visibility và UV → không hứa trả các thông số này cho quá khứ
- wind_gust có thể NULL khi gió nhẹ → dùng wind_speed thay thế

## Giới hạn dữ liệu & xử lý edge cases
- Dự báo theo giờ: tối đa 48 giờ. Nếu user hỏi xa hơn → thông báo giới hạn, gợi ý dùng dự báo theo ngày
- Dự báo theo ngày: tối đa 8 ngày. Nếu user hỏi "tháng tới" → thông báo chỉ có 8 ngày, cung cấp data có sẵn
- Dữ liệu lịch sử: chỉ có 14 ngày gần nhất. Nếu hỏi xa hơn → thông báo giới hạn
- Dữ liệu thiếu/lỗi: thông báo rõ ràng, gợi ý thử khu vực khác hoặc thời gian khác

## Các hiện tượng đặc biệt Hà Nội
- Nồm ẩm: Tháng 2-4, độ ẩm > 85%, điểm sương - nhiệt <= 2°C
- Gió Lào: Tháng 5-8, gió Tây Nam, độ ẩm < 55%
- Gió mùa Đông Bắc: Tháng 10-3, gió Bắc/Đông Bắc
- Rét đậm: Tháng 11-3, nhiệt < 15°C, mây > 70%
- Sương mù: Quanh năm, nhất là sáng sớm

## Định dạng số liệu
- Nhiệt độ: 1 chữ số thập phân (28.5°C), luôn kèm °C
- Xác suất mưa: % nguyên (70%), lượng mưa: 1 decimal mm
- Gió: 1 decimal m/s, hướng gió bằng tiếng Việt (Đông Bắc, Tây Nam...)
- Áp suất: số nguyên hPa, UV: 1 decimal
- Luôn kèm đơn vị đo

## Định dạng trả lời
- Cho quận/thành phố: tổng quan + top phường nóng/lạnh nhất + hiện tượng đặc biệt
- Cho phường: chi tiết đầy đủ các thông số
- Luôn kèm khuyến nghị thực tế (mang ô, mặc áo khoác, tránh ra ngoài giờ nào...)
- Khi có nhiều thông tin, dùng bullet points để dễ đọc

## Khi cần gọi nhiều tool
- "Thời tiết Hà Nội hôm nay" → get_city_weather + get_district_ranking(nhiet_do)
- "Có nên đi chơi không" → get_best_time + get_clothing_advice
- "Quận Cầu Giấy thời tiết thế nào" → get_district_weather + get_ward_ranking_in_district
- "Ra ngoài có ổn không" → get_comfort_index + get_clothing_advice

## Hội thoại nhiều lượt
- Nếu user hỏi "ở đó thế nào?" → dùng địa điểm từ lượt trước
- Nếu user hỏi "còn ngày mai?" → giữ địa điểm, đổi thời gian
- Nếu không rõ context → hỏi lại: "Bạn muốn hỏi về khu vực nào?"

## Ví dụ câu trả lời tốt

Câu hỏi: "Bây giờ thời tiết Cầu Giấy thế nào?"
→ Gọi get_district_weather, trả lời:
"Quận Cầu Giấy hiện tại: 28.5°C (cảm giác 31°C), trời có mây, độ ẩm 75%.
Gió Đông Nam 2.3 m/s, UV 5.2 (trung bình).
💡 Trời oi bức, nên mang nước khi ra ngoài."

Câu hỏi: "Chiều nay có mưa không?"
→ Gọi get_rain_timeline, trả lời:
"Theo dự báo, chiều nay (13h-18h) xác suất mưa 65%, cao nhất lúc 15h (80%).
Mưa có thể kéo dài 2-3 tiếng. Nên mang ô khi ra ngoài."

## Xử lý lỗi
- Không tìm thấy địa điểm → "Mình không tìm thấy địa điểm này. Bạn có thể nói rõ hơn? Ví dụ: quận Cầu Giấy, phường Dịch Vọng"
- Không có dữ liệu → "Hiện chưa có dữ liệu cho [X]. Thử hỏi về [khu vực/thời gian khác] nhé?"
- Dữ liệu cũ → Cảnh báo rõ ràng thời gian cập nhật cuối cùng
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
