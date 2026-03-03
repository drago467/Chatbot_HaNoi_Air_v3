""""
Batch Evaluation Script for Weather Chatbot.

Runs test queries and logs results for evaluation.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.agent import run_agent
from app.agent.evaluation_logger import get_evaluation_logger


def load_test_queries(csv_path):
    queries = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    return queries


def evaluate_query(question, expected_tool=None, expected_location=None):
    start_time = time.time()
    
    try:
        result = run_agent(message=question, thread_id="evaluation")
        
        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "expected_tool": expected_tool,
            "expected_location": expected_location,
            "response": response,
            "response_time_ms": elapsed_ms,
            "success": True,
            "error": None
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "question": question,
            "expected_tool": expected_tool,
            "expected_location": expected_location,
            "response": "",
            "response_time_ms": elapsed_ms,
            "success": False,
            "error": str(e)
        }


def run_evaluation(output_dir="data/evaluation"):
    logger = get_evaluation_logger(output_dir)
    
    test_file = Path(output_dir) / "hanoi_weather_chatbot_eval_questions.csv"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    queries = load_test_queries(str(test_file))
    print(f"Loaded {len(queries)} test queries")
    
    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Running: {q['question'][:50]}...")
        
        result = evaluate_query(
            question=q.get("question", q.get("query", "")),
            expected_tool=q.get("intent"),
            expected_location=q.get("location_name")
        )
        
        results.append(result)
        
        logger.log_conversation(
            session_id="batch_evaluation",
            turn_number=i,
            user_query=q["question"],
            llm_response=result["response"][:500],
            response_time_ms=result["response_time_ms"],
            error_type=result["error"]
        )
    
    successful = sum(1 for r in results if r["success"])
    avg_time = sum(r["response_time_ms"] for r in results) / len(results)
    
    print()
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries: {len(results)}")
    print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Failed: {len(results) - successful}")
    print(f"Avg response time: {avg_time:.0f}ms")
    
    output_file = Path(output_dir) / "evaluation_results.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate weather chatbot")
    parser.add_argument("--output", default="data/evaluation")
    args = parser.parse_args()
    
    run_evaluation(args.output)
