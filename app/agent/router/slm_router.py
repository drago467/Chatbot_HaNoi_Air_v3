"""SLM Router — classify user query → {intent, scope, confidence, rewritten_query?}.

Inference backend: Ollama HTTP API (primary).
Model: fine-tuned Qwen2.5-1.5B-Instruct / Qwen3-4B (GGUF Q4_K_M/Q8_0).

Multi-task output (Module 1b): when context is provided from ConversationState,
the model can also output rewritten_query for standalone contextual resolution.

Calibration (Module 4): applies temperature scaling and per-intent thresholds.
Smart routing (v4): safety-aware heuristics for ambiguous intent pairs.
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
    CALIBRATION_TEMPERATURE,
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

    New in multi-task mode:
        rewritten_query: Standalone query with resolved context (or None if not needed)
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
        """Return rewritten_query if available, else None (caller uses original)."""
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

    Supports:
    - Multi-task output: routing + contextual query rewriting in 1 call (Module 1b)
    - Temperature scaling calibration for confidence (Module 4)
    - Per-intent adaptive thresholds (Module 4)
    """

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        confidence_threshold: float = 0.75,
        timeout: float = 10.0,
        calibration_temperature: float = CALIBRATION_TEMPERATURE,
        per_intent_thresholds: dict | None = None,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.calibration_temperature = calibration_temperature
        self.per_intent_thresholds = per_intent_thresholds or PER_INTENT_THRESHOLDS
        self._client = httpx.Client(timeout=timeout)
        self._healthy: bool | None = None

    def _apply_calibration(self, confidence: float) -> float:
        """Apply temperature scaling calibration to raw confidence.

        When T > 1: softens (reduces) confidence → more conservative routing
        When T < 1: sharpens confidence → more aggressive routing
        When T = 1: identity (no change)
        """
        if abs(self.calibration_temperature - 1.0) < 0.001:
            return confidence  # No-op
        # Approximate: calibrated_conf = conf^(1/T) normalized
        # (Simplified scalar approximation of temperature scaling on max prob)
        import math
        if confidence <= 0 or confidence >= 1:
            return confidence
        try:
            # log-odds rescaling
            logit = math.log(confidence / (1 - confidence))
            scaled_logit = logit / self.calibration_temperature
            return 1.0 / (1.0 + math.exp(-scaled_logit))
        except (ValueError, OverflowError):
            return confidence

    def _get_threshold(self, intent: str) -> float:
        """Get per-intent threshold, fallback to global threshold."""
        return self.per_intent_thresholds.get(intent, self.confidence_threshold)

    # ── Smart Routing Heuristics (v4) ──
    # Safety-aware keyword checks to correct common model confusion pairs

    # Keywords strongly indicating weather_alert (safety-critical)
    _ALERT_KEYWORDS = re.compile(
        r"\b(bão|lũ|lu|ngập|ngap|giông|giong|sét|set|lốc|loc"
        r"|cảnh báo|canh bao|nguy hiểm|nguy hiem"
        r"|mưa to|mua to|mưa lớn|mua lon|mưa đá|mua da"
        r"|áp thấp|ap thap|rét hại|ret hai|nắng nóng gay gắt)\b",
        re.IGNORECASE,
    )

    # Keywords strongly indicating temperature_query (not general weather)
    _TEMP_KEYWORDS = re.compile(
        r"(nhiệt độ|nhiet do|bao nhiêu độ|bao nhieu do|mấy độ|may do"
        r"|\bnóng\b.*\bkhông\b|\blạnh\b.*\bkhông\b"
        r"|\bnong\b.*\bkhong\b|\blanh\b.*\bkhong\b)",
        re.IGNORECASE,
    )

    # Keywords strongly indicating current_weather (general overview, not specific param)
    _WEATHER_OVERVIEW_KEYWORDS = re.compile(
        r"(thời tiết|thoi tiet).*(thế nào|the nao|sao|ra sao)",
        re.IGNORECASE,
    )

    def _apply_smart_routing(
        self, query: str, intent: str, confidence: float
    ) -> tuple[str, float]:
        """Apply keyword-based heuristics to correct known confusion pairs.

        Returns (corrected_intent, adjusted_confidence).

        Safety principle:
        - weather_alert is NEVER downgraded (miss storm warning > false alert)
        - temperature_query vs current_weather resolved by keyword signal
        """
        q = query.lower()

        # Rule 1: Safety — upgrade to weather_alert if danger keywords detected
        if intent != "weather_alert" and self._ALERT_KEYWORDS.search(q):
            # Don't override if it's clearly a "dự báo" (forecast) query
            if "dự báo" not in q and "du bao" not in q:
                logger.debug("Smart routing: %s → weather_alert (safety keywords)", intent)
                return "weather_alert", max(confidence, 0.80)

        # Rule 1b: Telex "bao" = "bão" (storm) — only when standalone,
        # not in "bao nhieu/bao nhiêu/du bao/bao gio/bao giờ/bao lau/bao lâu"
        if intent != "weather_alert" and re.search(r"\bbao\b", q):
            if not re.search(r"(du bao|dự báo|bao nhi|bao gi|bao l[aâ]u)", q):
                logger.debug("Smart routing: %s → weather_alert (telex 'bao'='bão')", intent)
                return "weather_alert", max(confidence, 0.75)

        # Rule 2: Resolve current_weather ↔ temperature_query
        if intent == "current_weather" and self._TEMP_KEYWORDS.search(q):
            logger.debug("Smart routing: current_weather → temperature_query (temp keywords)")
            return "temperature_query", confidence
        if intent == "temperature_query" and self._WEATHER_OVERVIEW_KEYWORDS.search(q):
            logger.debug("Smart routing: temperature_query → current_weather (overview keywords)")
            return "current_weather", confidence

        return intent, confidence

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
        Note: Anaphora no longer causes immediate fallback — context is passed to model.
        """
        t0 = time.perf_counter()

        # Build user message: query + optional context for multi-task rewriting
        user_message = self._build_user_message(query, context)

        # Call Ollama
        try:
            raw_text = self._call_ollama(user_message)
        except Exception as e:
            logger.warning("SLM Router error: %s", e)
            return RouterResult(
                latency_ms=_elapsed_ms(t0),
                fallback_reason=f"model_error: {e}",
            )

        # Parse JSON response
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

        # Confidence: use model-reported if available, else 1.0 for valid JSON
        raw_confidence = parsed.get("confidence", 1.0)
        if not isinstance(raw_confidence, (int, float)):
            raw_confidence = 1.0
        raw_confidence = float(raw_confidence)

        # Apply temperature scaling calibration
        confidence = self._apply_calibration(raw_confidence)

        # Apply smart routing heuristics (v4: safety-aware intent correction)
        intent, confidence = self._apply_smart_routing(query, intent, confidence)

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
        """Build user message for Ollama, injecting context if available."""
        if context is None or context.turn_count == 0:
            return query

        # Inject context as JSON prefix so model can rewrite if needed
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
            "options": {
                "temperature": 0.0,
                "num_predict": 128,  # Increased for rewritten_query field
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
        """Parse JSON from model output. Handles extra text around JSON."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from text
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def health_check(self) -> bool:
        """Check if Ollama is reachable and model is loaded."""
        try:
            resp = self._client.get(f"{self.ollama_base_url}/api/tags")
            if resp.status_code != 200:
                self._healthy = False
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # Check if our model exists (with or without :latest tag)
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
    """Get or create the singleton SLM Router.

    Loads calibration.json if it exists (temperature + per-intent thresholds).
    Falls back to config defaults if file not found or invalid.
    """
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                from app.agent.router.config import (
                    CALIBRATION_FILE,
                    CALIBRATION_TEMPERATURE,
                    CONFIDENCE_THRESHOLD,
                    OLLAMA_BASE_URL,
                    OLLAMA_MODEL,
                    PER_INTENT_THRESHOLDS,
                )

                # Try to load fitted calibration from file
                cal_temperature = CALIBRATION_TEMPERATURE
                per_intent_thresholds = dict(PER_INTENT_THRESHOLDS)
                try:
                    if CALIBRATION_FILE.exists():
                        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                            cal = json.load(f)
                        if "temperature" in cal and cal["temperature"] != 1.0:
                            cal_temperature = float(cal["temperature"])
                            logger.info("Loaded calibration T=%.4f from %s", cal_temperature, CALIBRATION_FILE)
                        if cal.get("per_intent_thresholds"):
                            per_intent_thresholds.update(cal["per_intent_thresholds"])
                            logger.info("Loaded per-intent thresholds from calibration file")
                except Exception as e:
                    logger.warning("Could not load calibration file: %s", e)

                _router = SLMRouter(
                    ollama_base_url=OLLAMA_BASE_URL,
                    model=OLLAMA_MODEL,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    calibration_temperature=cal_temperature,
                    per_intent_thresholds=per_intent_thresholds,
                )
    return _router


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
