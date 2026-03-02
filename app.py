"""
HanoiAir Weather Chatbot - Streamlit UI

A clean, ChatGPT-like interface for the weather chatbot.
"""
import streamlit as st
import time
import uuid
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Chatbot Thời Tiết Hà Nội",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "location" not in st.session_state:
    st.session_state.location = None


def get_weather_icon(weather_main: str) -> str:
    """Get weather icon based on weather condition."""
    icons = {
        "Clear": "sunny",
        "Clouds": "cloudy",
        "Rain": "rainy",
        "Drizzle": "partly_cloudy_rainy",
        "Thunderstorm": "thunderstorm",
        "Snow": "ac_unit",
        "Mist": "foggy",
        "Fog": "foggy",
    }
    return icons.get(weather_main, "sunny")


def call_agent(prompt: str, thread_id: str):
    """Call the agent and yield response chunks."""
    from app.agent.agent import stream_agent
    
    try:
        for chunk in stream_agent(prompt, thread_id=thread_id):
            yield chunk
    except Exception as e:
        yield f"Loi: {str(e)}"


def get_current_weather_summary():
    """Get quick weather summary for sidebar."""
    from app.db.dal import query
    
    try:
        result = query("""
            SELECT temp, humidity, weather_main, wind_speed
            FROM fact_weather_hourly
            WHERE ward_id = 'ID_00364'
            ORDER BY ts_utc DESC
            LIMIT 1
        """)
        
        if result:
            return result[0]
    except:
        pass
    return None


# Sidebar
with st.sidebar:
    st.title("Thoi Tiet Ha Noi")
    
    weather = get_current_weather_summary()
    if weather:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nhiet do", f"{weather.get('temp', '--')}C")
        with col2:
            st.metric("Do am", f"{weather.get('humidity', '--')}%")
    
    st.divider()
    
    st.subheader("Chon dia diem")
    
    from app.db.dal import query
    try:
        districts = query("""
            SELECT DISTINCT district_name_vi 
            FROM dim_ward 
            ORDER BY district_name_vi
        """)
        
        district_names = [d["district_name_vi"] for d in districts if d.get("district_name_vi")]
        selected_district = st.selectbox("Quan/Huyen", district_names, index=0)
        
        wards = query("""
            SELECT ward_id, ward_name_vi 
            FROM dim_ward 
            WHERE district_name_vi = %s
            ORDER BY ward_name_vi
        """, (selected_district,))
        
        ward_names = {w["ward_name_vi"]: w["ward_id"] for w in wards}
        selected_ward = st.selectbox("Phuong/Xa", list(ward_names.keys()))
        
        if selected_ward:
            st.session_state.location = ward_names[selected_ward]
    except Exception as e:
        st.warning(f"Khong the tai danh sach: {e}")
    
    st.divider()
    
    if st.button("Xoa hoi thoai", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# Main chat area
st.title("Chatbot Thoi Tiet Ha Noi")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hoi ve thoi tiet Ha Noi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.status("Dang xu ly...", expanded=True):
            start_time = time.time()
            
            for chunk in call_agent(prompt, st.session_state.thread_id):
                full_response += chunk
                message_placeholder.markdown(full_response + "|")
            
            elapsed = time.time() - start_time
            st.caption(f"Thoi gian: {elapsed:.1f}s")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })


st.markdown("---")
st.caption("Goi y: 'Thoi tiet hom nay the nao?', 'Ngay mai co mua khong?', 'So sanh Cau Giay va Hoan Kiem'")
