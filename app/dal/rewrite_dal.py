"""Question rewriting DAL - Extract location and intent from user questions.

Uses LLM to rewrite questions and extract location information.
This is a best practice pattern from LangGraph for handling ambiguous queries.
"""

import json
import re
import os
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def get_llm():
    """Get LLM instance for question rewriting."""
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    return ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url=os.getenv("API_BASE"),
        api_key=os.getenv("API_KEY")
    )


REWRITE_PROMPT = """Phân tích câu hỏi và trả lời JSON.

Câu hỏi: {question}

Trả lời CHỈ JSON, không giải thích:
{{"location": "tên địa điểm", "district": "quận/huyện", "ward": "phường/xã", "is_ambiguous": true/false, "confidence": 0.0-1.0}}"""


def get_districts_list() -> str:
    """Get list of districts from database."""
    from app.db.dal import query
    
    results = query("""
        SELECT DISTINCT district_name_vi 
        FROM dim_ward 
        WHERE district_name_vi IS NOT NULL
        ORDER BY district_name_vi
    """)
    
    return "\n".join([r["district_name_vi"] for r in results])


def get_wards_list() -> str:
    """Get list of wards from database."""
    from app.db.dal import query
    
    results = query("""
        SELECT ward_name_vi, district_name_vi 
        FROM dim_ward 
        WHERE district_name_vi IS NOT NULL
        ORDER BY district_name_vi, ward_name_vi
        LIMIT 100
    """)
    
    lines = []
    for r in results:
        lines.append(f"{r['ward_name_vi']}, {r['district_name_vi']}")
    
    return "\n".join(lines)


def rewrite_question(question: str) -> Dict[str, Any]:
    """Rewrite question and extract location using LLM."""
    try:
        llm = get_llm()
        
        prompt = REWRITE_PROMPT.format(question=question)
        
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Try to parse JSON from response
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text (handles markdown code blocks)
        try:
            # Remove markdown code blocks
            cleaned = content.replace('```json', '').replace('```', '').strip()
            # Find JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start >= 0 and end > start:
                json_str = cleaned[start:end+1]
                result = json.loads(json_str)
                return result
        except:
            pass
        
        # If all parsing fails, return error
        return {
            "location": None,
            "district": None,
            "ward": None,
            "is_ambiguous": True,
            "confidence": 0.0,
            "reasoning": "LLM response parsing failed"
        }
        
    except Exception as e:
        return {
            "location": None,
            "district": None,
            "ward": None,
            "is_ambiguous": True,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def resolve_with_rewrite(location_hint: str) -> Dict[str, Any]:
    """Resolve location using question rewriting + database lookup."""
    from app.dal.location_dal import resolve_location
    
    rewrite_result = rewrite_question(location_hint)
    
    if rewrite_result.get("is_ambiguous") or rewrite_result.get("confidence", 0) < 0.5:
        return {
            "status": "ambiguous",
            "level": "unknown",
            "needs_clarification": True,
            "rewrite_result": rewrite_result,
            "suggestion": f"Địa điểm '{location_hint}' không rõ ràng. " +
                         "Bạn có thể nói rõ hơn không? Ví dụ: 'quận Cầu Giấy' hoặc 'phường Xuân Đỉnh'"
        }
    
    district = rewrite_result.get("district")
    if district:
        district_result = resolve_location(district)
        if district_result.get("status") in ("exact", "fuzzy"):
            return {
                "status": "ok",
                "level": "district",
                "district_name": district_result.get("data", {}).get("district_name_vi"),
                "rewrite_result": rewrite_result
            }
    
    result = resolve_location(location_hint)
    
    if result.get("status") in ("exact", "fuzzy"):
        return {
            "status": "ok",
            "level": result.get("level"),
            "district_name": result.get("data", {}).get("district_name_vi"),
            "ward_id": result.get("data", {}).get("ward_id"),
            "rewrite_result": rewrite_result
        }
    
    return {
        "status": "ambiguous",
        "level": "unknown",
        "needs_clarification": True,
        "rewrite_result": rewrite_result,
        "suggestion": f"Không tìm thấy dịa điểm '{location_hint}'. " +
                     "Vui lòng cho biết tên phường/xã hoặc quận/huyện."
    }
