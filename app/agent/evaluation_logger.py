"""
Evaluation Logger for Weather Chatbot.

Logs conversations and tool calls for evaluation purposes.
"""
import csv
import os
from datetime import datetime
from app.dal.timezone_utils import now_ict
from typing import Optional, List, Dict, Any
from pathlib import Path


class EvaluationLogger:
    """Logger for evaluating chatbot performance."""
    
    def __init__(self, log_dir: str = "data/evaluation"):
        """Initialize logger with log directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversations_file = self.log_dir / "conversations.csv"
        self.tool_calls_file = self.log_dir / "tool_calls.csv"
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if not self.conversations_file.exists():
            with open(self.conversations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'turn_number', 'user_query',
                    'resolved_location', 'llm_response', 'response_time_ms',
                    'tool_calls', 'error_type', 'user_rating'
                ])
        
        if not self.tool_calls_file.exists():
            with open(self.tool_calls_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'turn_number', 'tool_name',
                    'tool_input', 'tool_output', 'success', 'execution_time_ms'
                ])
    
    def log_conversation(
        self,
        session_id: str,
        turn_number: int,
        user_query: str,
        resolved_location: Optional[str] = None,
        llm_response: str = "",
        response_time_ms: float = 0,
        tool_calls: Optional[List[Dict]] = None,
        error_type: Optional[str] = None,
        user_rating: Optional[int] = None
    ):
        """Log a conversation turn."""
        with open(self.conversations_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                now_ict().isoformat(),
                session_id,
                turn_number,
                user_query,
                resolved_location or "",
                llm_response[:500],  # Truncate long responses
                round(response_time_ms, 2),
                str(tool_calls) if tool_calls else "",
                error_type or "",
                user_rating or ""
            ])
    
    def log_tool_call(
        self,
        session_id: str,
        turn_number: int,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool = True,
        execution_time_ms: float = 0
    ):
        """Log a tool call."""
        with open(self.tool_calls_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                now_ict().isoformat(),
                session_id,
                turn_number,
                tool_name,
                str(tool_input)[:200],  # Truncate
                str(tool_output)[:200] if tool_output else "",
                success,
                round(execution_time_ms, 2)
            ])
    
    def get_conversations(self) -> List[Dict]:
        """Load all logged conversations."""
        if not self.conversations_file.exists():
            return []
        
        conversations = []
        with open(self.conversations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conversations.append(row)
        return conversations
    
    def get_tool_calls(self) -> List[Dict]:
        """Load all logged tool calls."""
        if not self.tool_calls_file.exists():
            return []
        
        tool_calls = []
        with open(self.tool_calls_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tool_calls.append(row)
        return tool_calls
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics from logged data."""
        conversations = self.get_conversations()
        tool_calls = self.get_tool_calls()
        
        if not conversations:
            return {"error": "No data"}
        
        # Basic metrics
        total_turns = len(conversations)
        total_sessions = len(set(c['session_id'] for c in conversations))
        
        # Response time
        response_times = [
            float(c['response_time_ms']) 
            for c in conversations 
            if c['response_time_ms']
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Error rate
        errors = [c for c in conversations if c['error_type']]
        error_rate = len(errors) / total_turns if total_turns > 0 else 0
        
        # Tool call success rate
        successful_tools = [t for t in tool_calls if t['success'] == 'True']
        tool_success_rate = (
            len(successful_tools) / len(tool_calls) 
            if tool_calls else 0
        )
        
        return {
            "total_turns": total_turns,
            "total_sessions": total_sessions,
            "avg_response_time_ms": round(avg_response_time, 2),
            "error_rate": round(error_rate * 100, 2),
            "tool_success_rate": round(tool_success_rate * 100, 2),
            "total_tool_calls": len(tool_calls)
        }


# Global logger instance
_logger: Optional[EvaluationLogger] = None


def get_evaluation_logger(log_dir: str = "data/evaluation") -> EvaluationLogger:
    """Get global evaluation logger instance.
    
    Args:
        log_dir: Directory to store evaluation logs. Defaults to data/evaluation.
    """
    global _logger
    if _logger is None:
        _logger = EvaluationLogger(log_dir=log_dir)
    return _logger
