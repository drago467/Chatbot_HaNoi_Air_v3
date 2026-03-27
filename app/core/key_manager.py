from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class _KeyState:
    """State for a single API key."""
    key: str
    name: str
    cooldown_until: float = 0.0
    minute_window_start: float = 0.0
    minute_count: int = 0
    disabled_services: Set[str] = field(default_factory=set)


class OpenWeatherKeyManager:
    """Unified API key manager for OpenWeather with 5-key pool.

    All 5 keys (0-4) can be used for:
    - One Call 3.0 API (weather current, forecast, timemachine)

    Features:
    - Round-robin across all available keys
    - Per-key rate limiting (60 req/min per key)
    - Per-key cooldown on 429 errors
    - Per-service blacklisting on 401 errors

    Env vars:
    - OPENWEATHER_API_KEY_0
    - OPENWEATHER_API_KEY_1
    - OPENWEATHER_API_KEY_2
    - OPENWEATHER_API_KEY_3
    - OPENWEATHER_API_KEY_4
    """

    def __init__(
        self,
        per_minute_limit: int = 60,
        cooldown_seconds: int = 120,
    ) -> None:
        self.per_minute_limit = per_minute_limit
        self.cooldown_seconds = cooldown_seconds

        # Load all 5 keys into a unified pool
        self._states: List[_KeyState] = []
        for i in range(5):
            k = os.getenv(f"OPENWEATHER_API_KEY_{i}")
            if k:
                self._states.append(_KeyState(key=k.strip("'\" "), name=f"key_{i}"))

        if not self._states:
            raise ValueError(
                "No OpenWeather API keys found. Expected OPENWEATHER_API_KEY_0..4"
        )
        
        self._rr_index = 0
        self._lock = threading.Lock()
        logger.info(f"Loaded {len(self._states)} API keys into unified pool")

    def _reset_minute_window_if_needed(self, st: _KeyState, now: float) -> None:
        """Reset minute counter if we've crossed into a new minute."""
        if st.minute_window_start == 0.0 or (now - st.minute_window_start) >= 60.0:
            st.minute_window_start = now
            st.minute_count = 0

    def _is_available(self, st: _KeyState, now: float, service: str) -> bool:
        """Check if a key is available for the given service."""
        # Check if blacklisted for this service
        if service in st.disabled_services:
            return False
        # Check cooldown
        if now < st.cooldown_until:
            return False
        # Check minute limit
        self._reset_minute_window_if_needed(st, now)
        return st.minute_count < self.per_minute_limit

    def get_key(self, service: str = "onecall") -> str:
        """Get an available key for the requested service.

        Round-robins through all keys until finding one that:
        - Is not on cooldown
        - Has not hit per-minute limit
        - Is not blacklisted for this service

        Args:
            service: 'onecall' or 'timemachine'

        Returns:
            An available API key string

        Raises:
            RuntimeError: If no keys are available (message includes wait time)
        """
        with self._lock:
            now = time.time()
            n = len(self._states)
            best_wait_s: Optional[float] = None

            for i in range(n):
                idx = (self._rr_index + i) % n
                st = self._states[idx]

                if self._is_available(st, now, service):
                    # Found an available key
                    self._rr_index = (idx + 1) % n
                    st.minute_count += 1
                    return st.key

            # No key available — compute minimum wait across ALL keys
            for st in self._states:
                if service in st.disabled_services:
                    continue  # blacklisted, skip
                # Time until cooldown expires
                wait_s = max(0.0, st.cooldown_until - now)
                if wait_s <= 0:
                    # Not on cooldown — wait for minute window to reset
                    self._reset_minute_window_if_needed(st, now)
                    if st.minute_count >= self.per_minute_limit:
                        wait_s = max(0.0, 60.0 - (now - st.minute_window_start))
                    else:
                        wait_s = 0.0  # actually available (race condition)
                if best_wait_s is None or wait_s < best_wait_s:
                    best_wait_s = wait_s

        raise RuntimeError(
            f"All {n} keys exhausted for '{service}' "
            f"(limit={self.per_minute_limit}/min per key). "
            f"Try again in ~{(best_wait_s or 0.0):.1f}s"
        )

    def get_wait_seconds(self, service: str = "onecall") -> float:
        """Return estimated seconds until at least one key becomes available.

        Returns 0.0 if a key is already available.
        """
        with self._lock:
            now = time.time()
            best_wait: Optional[float] = None

            for st in self._states:
                if service in st.disabled_services:
                    continue
                # Check cooldown
                if now < st.cooldown_until:
                    wait = st.cooldown_until - now
                else:
                    # Check minute limit
                    self._reset_minute_window_if_needed(st, now)
                    if st.minute_count < self.per_minute_limit:
                        return 0.0  # this key is available right now
                    wait = max(0.0, 60.0 - (now - st.minute_window_start))

                if best_wait is None or wait < best_wait:
                    best_wait = wait

            return best_wait if best_wait is not None else 60.0

    def report_failure(self, key: str, status_code: int, service: str) -> None:
        """Handle API failure by status code.

        - 429: Put key on cooldown
        - 401: Blacklist key for this service only
        """
        now = time.time()
        st = next((s for s in self._states if s.key == key), None)

        if not st:
            return

        if status_code == 429:
            st.cooldown_until = max(st.cooldown_until, now + self.cooldown_seconds)
            logger.warning(
                f"Key {st.name} hit rate limit (429) for '{service}'. "
                f"Cooldown: {self.cooldown_seconds}s"
            )
        elif status_code == 401:
            st.disabled_services.add(service)
            logger.error(
                f"Key {st.name} is UNAUTHORIZED (401) for '{service}'. "
                f"Blacklisted for this service."
            )

    def report_rate_limited(self, key: str) -> None:
        """Legacy helper for 429."""
        self.report_failure(key, 429, "unknown")

    def report_success(self, key: str) -> None:
        """Optional hook for future metrics."""
        _ = key

    def debug_state(self) -> List[Dict]:
        """Return a JSON-serializable snapshot of internal state."""
        now = time.time()
        out: List[Dict] = []
        for st in self._states:
            self._reset_minute_window_if_needed(st, now)
            out.append({
                "name": st.name,
                    "cooldown_remaining_s": max(0.0, st.cooldown_until - now),
                "minute_count": st.minute_count,
                "minute_limit": self.per_minute_limit,
                "disabled_services": list(st.disabled_services),
            })
        return out

    def get_available_count(self, service: str) -> int:
        """Return count of keys currently available for a service."""
        now = time.time()
        return sum(1 for st in self._states if self._is_available(st, now, service))