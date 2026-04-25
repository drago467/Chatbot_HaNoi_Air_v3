# HanoiWeather Chatbot — Makefile
#
# Tiện dụng cho demo + bảo vệ khóa luận: một lệnh chạy cả stack.
# Windows: cần WSL/Git Bash (đã có sẵn nếu dùng Claude Code hoặc git for windows).
# Linux/macOS: chạy trực tiếp.

PYTHON := python
PIP := pip
API_PORT := 8000
UI_PORT := 8501

.PHONY: help install run-api run-ui run-all test eval ingest clean

help:
	@echo "HanoiWeather Chatbot — make targets:"
	@echo ""
	@echo "  make install    — pip install dependencies"
	@echo "  make run-api    — Khởi động FastAPI (port $(API_PORT))"
	@echo "  make run-ui     — Khởi động Streamlit UI (port $(UI_PORT))"
	@echo "  make run-all    — Chạy cả API + UI song song (cần 2 terminal hoặc &)"
	@echo "  make test       — Chạy pytest full suite"
	@echo "  make eval       — Chạy eval 199 câu → data/evaluation/traces/"
	@echo "  make ingest     — Ingest weather mới nhất từ OpenWeather"
	@echo "  make clean      — Xóa __pycache__, .pytest_cache"
	@echo ""

install:
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "	  Dependencies installed."
	@echo "   Nhớ cp env_example.txt .env rồi điền API keys."
	@echo "   Nếu dùng SLM router: ollama pull hanoi-weather-router"

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

run-ui:
	streamlit run app.py --server.port $(UI_PORT)

run-all:
	@echo "Khởi động API + UI song song..."
	@echo "API:  http://localhost:$(API_PORT)/docs"
	@echo "UI:   http://localhost:$(UI_PORT)"
	@echo "Ctrl+C để dừng."
	uvicorn app.api.main:app --host 0.0.0.0 --port $(API_PORT) & \
	streamlit run app.py --server.port $(UI_PORT); \
	kill %1 2>/dev/null || true

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	@mkdir -p data/evaluation/traces
	$(PYTHON) -m scripts.eval.run_traces \
		--input data/evaluation/hanoi_weather_chatbot_eval_questions.csv \
		--output data/evaluation/traces/full_run_$$(date +%Y%m%d_%H%M).jsonl

ingest:
	$(PYTHON) -m app.scripts.ingest_openweather_async

ingest-history:
	$(PYTHON) -m app.scripts.ingest_openweather_async --days 7

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache 2>/dev/null || true
	@echo " Cleaned __pycache__ + .pytest_cache"
