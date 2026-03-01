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

## Vi du
User: "Cau Giay hom nay the nao?"
Bot: "Cau Giay hien 28C, troi nang. 
Luu y: Nhiet do cao hon trung binh thang 3 6C - day la ngay am bat thuong.
Neu ban di chay bo, nen tap buoi sang som (truoc 7h) de tranh nang nong."

User: "Co canh bao gi khong?"
Bot: "Hien khong co canh bao nguy hiem. Tuy nhien, chieu nay co kha nang mua 40%, nen mang theo o neu di ra ngoai."
"""


def create_weather_agent():
    """Create weather agent with LangGraph."""
    
    # Model - use proxy API
    API_BASE = os.getenv("API_BASE")
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not API_BASE or not API_KEY:
        raise ValueError("API_BASE and OPENAI_API_KEY must be set in .env")
    
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        base_url=API_BASE,
        api_key=API_KEY
    )
    
    # Checkpointer - use Postgres for persistence
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")
    
    checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
    
    # Create agent
    agent = create_react_agent(
        model=model,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return agent


def run_agent(message: str, thread_id: str = "default") -> dict:
    """Run agent with a message.
    
    Args:
        message: User message
        thread_id: Thread ID for conversation persistence
        
    Returns:
        Agent response
    """
    agent = create_weather_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    
    return result


# For testing
if __name__ == "__main__":
    print("Testing agent...")
    
    # Test tool imports
    from app.agent.tools import TOOLS
    print(f"Loaded {len(TOOLS)} tools:")
    for t in TOOLS:
        print(f"  - {t.name}")
    
    print("\nAgent ready!")
