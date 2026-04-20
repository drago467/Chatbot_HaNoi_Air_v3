"""SLM Router — classify user query → {intent, scope, confidence, rewritten_query?}.

Inference backend: Ollama HTTP API (primary).
Model: fine-tuned Qwen3-1.4B-Instruct / Qwen3-4B-v6 (GGUF Q4_K_M/Q8_0).

Multi-task output: when context is provided from ConversationState, the model
can also output rewritten_query for standalone contextual resolution.

Calibration đã bị remove — model hard-code output confidence=0.9 (training data
artifact), temperature scaling T=1.4019 chỉ làm confidence cluster 0.83 giả tạo.
Giờ dùng raw confidence + per-intent threshold.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

from typing import TYPE_CHECKING

import httpx

from app.agent.router.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PER_INTENT_THRESHOLDS,
    ROUTER_SYSTEM_PROMPT,
    VALID_INTENTS,
    VALID_SCOPES,
)

if TYPE_CHECKING:
    from app.agent.conversation_state import ConversationState

logger = logging.getLogger(__name__)


class RouterResult:
    """Classification result from SLM Router.

    Multi-task mode: rewritten_query chứa câu truy vấn đã resolve context
    (hoặc None nếu không cần).
    """

    __slots__ = ("intent", "scope", "confidence", "latency_ms", "fallback_reason", "rewritten_query")

    def __init__(
        self,
        intent: str = "",
        scope: str = "city",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        fallback_reason: str | None = None,
        rewritten_query: str | None = None,
    ):
        self.intent = intent
        self.scope = scope
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.fallback_reason = fallback_reason
        self.rewritten_query = rewritten_query

    @property
    def should_fallback(self) -> bool:
        return self.fallback_reason is not None

    @property
    def effective_query(self) -> str | None:
        """Trả về rewritten_query nếu có, else None (caller dùng query gốc)."""
        return self.rewritten_query if self.rewritten_query else None

    def __repr__(self) -> str:
        if self.fallback_reason:
            return f"RouterResult(FALLBACK: {self.fallback_reason}, {self.latency_ms:.0f}ms)"
        rewrite = f", rewrite='{self.rewritten_query[:40]}...'" if self.rewritten_query else ""
        return (
            f"RouterResult({self.intent}/{self.scope}, "
            f"conf={self.confidence:.2f}, {self.latency_ms:.0f}ms{rewrite})"
        )


class SLMRouter:
    """SLM-based intent + scope classifier using Ollama.

    Dùng raw confidence từ model output + per-intent threshold.
    """

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        confidence_threshold: float = 0.75,
        timeout: float = 30.0,
        per_intent_thresholds: dict | None = None,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.per_intent_thresholds = per_intent_thresholds or PER_INTENT_THRESHOLDS
        self._client = httpx.Client(timeout=timeout)
        self._healthy: bool | None = None

    def _get_threshold(self, intent: str) -> float:
        """Get per-intent threshold, fallback to global threshold."""
        return self.per_intent_thresholds.get(intent, self.confidence_threshold)

    def classify(
        self,
        query: str,
        context: "ConversationState | None" = None,
    ) -> RouterResult:
        """Classify user query → RouterResult.

        Args:
            query: User query to classify
            context: Optional ConversationState from previous turns.
                     When provided, enables multi-task output (routing + rewriting).

        Returns RouterResult with fallback_reason set if:
        - Model unavailable / error
        - JSON parse failure
        - Confidence below per-intent threshold
        - Invalid intent/scope
        """
        t0 = time.perf_counter()

        user_message = self._build_user_message(query, context)

        try:
            raw_text = self._call_ollama(user_message)
        except Exception as e:
            logger.warning("SLM Router error: %s", e)
            return RouterResult(
                latency_ms=_elapsed_ms(t0),
                fallback_reason=f"model_error: {e}",
            )

        parsed = self._parse_response(raw_text)
        if parsed is None:
            logger.warning("SLM Router JSON parse failed: %r", raw_text)
            return RouterResult(
                latency_ms=_elapsed_ms(t0),
                fallback_reason=f"json_parse_error: {raw_text[:100]}",
            )

        intent = parsed.get("intent", "")
        scope = parsed.get("scope", "city")
        rewritten_query = parsed.get("rewritten_query") or None

        # Validate intent/scope
        if intent not in VALID_INTENTS:
            return RouterResult(
                intent=intent,
                scope=scope,
                latency_ms=_elapsed_ms(t0),
                fallback_reason=f"invalid_intent: {intent}",
            )
        if scope not in VALID_SCOPES:
            scope = "city"  # safe default

        # Raw confidence từ model (không calibrate — xem docstring module)
        raw_confidence = parsed.get("confidence", 1.0)
        if not isinstance(raw_confidence, (int, float)):
            raw_confidence = 1.0
        confidence = float(raw_confidence)

        ms = _elapsed_ms(t0)

        # Per-intent threshold check
        threshold = self._get_threshold(intent)
        if confidence < threshold:
            return RouterResult(
                intent=intent,
                scope=scope,
                confidence=confidence,
                latency_ms=ms,
                rewritten_query=rewritten_query,
                fallback_reason=f"low_confidence: {confidence:.2f}",
            )

        return RouterResult(
            intent=intent,
            scope=scope,
            confidence=confidence,
            latency_ms=ms,
            rewritten_query=rewritten_query,
        )

    def _build_user_message(
        self, query: str, context: "ConversationState | None"
    ) -> str:
        """Build user message cho Ollama, inject context nếu có."""
        if context is None or context.turn_count == 0:
            return query

        ctx = context.to_context_json()
        context_str = json.dumps(ctx, ensure_ascii=False)
        return f"[CONTEXT: {context_str}]\n{query}"

    def _call_ollama(self, user_message: str) -> str:
        """Call Ollama /api/chat endpoint."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.0,
                "num_predict": 128,  # Increased cho rewritten_query field
            },
        }
        resp = self._client.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _parse_response(self, text: str) -> dict | None:
        """Parse JSON từ model output. Handle text thừa quanh JSON."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def health_check(self) -> bool:
        """Check if Ollama reachable and model loaded."""
        try:
            resp = self._client.get(f"{self.ollama_base_url}/api/tags")
            if resp.status_code != 200:
                self._healthy = False
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            self._healthy = any(
                self.model in m or m.startswith(self.model + ":") for m in models
            )
            return self._healthy
        except Exception:
            self._healthy = False
            return False

    def close(self):
        self._client.close()


# ── Singleton ──
_router: SLMRouter | None = None
_router_lock = threading.Lock()


def get_router() -> SLMRouter:
    """Get or create the singleton SLM Router."""
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                from app.agent.router.config import (
                    CONFIDENCE_THRESHOLD,
                    OLLAMA_BASE_URL,
                    OLLAMA_MODEL,
                    PER_INTENT_THRESHOLDS,
                )

                _router = SLMRouter(
                    ollama_base_url=OLLAMA_BASE_URL,
                    model=OLLAMA_MODEL,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    per_intent_thresholds=dict(PER_INTENT_THRESHOLDS),
                )
    return _router


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
