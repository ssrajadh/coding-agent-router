import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class RateLimitError(Exception):
    """Frontier backend returned 429 after all retries. Callers should fail-soft to local."""


class OllamaBackend:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
    async def chat_completion(self, body: dict) -> dict:
        payload = {**body, "model": self.model, "stream": False}
        payload.pop("stream_options", None)
        r = await self.client.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        r.raise_for_status()
        return r.json()


class _RetryableRateLimit(Exception):
    """Internal: thrown to drive tenacity backoff on 429. Converted to RateLimitError at the boundary."""


class NIMBackend:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(timeout=300.0, headers=self.headers)

    async def chat_completion(self, body: dict) -> dict:
        # Cap retries on 429 hard: 2 attempts with short backoff, then surface
        # RateLimitError so the proxy can fail-soft to local. This is the fix for
        # the all-frontier and full_system runs that ate their entire wall budget
        # on tenacity retries when NIM throttled (see SMOKE_TEST_REPORT.md).
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(min=1, max=4),
            retry=retry_if_exception_type(_RetryableRateLimit),
            reraise=True,
        )
        async def _attempt() -> dict:
            payload = {**body, "model": self.model, "stream": False}
            payload.pop("stream_options", None)
            r = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
            if r.status_code == 429:
                raise _RetryableRateLimit("nim 429")
            if r.status_code >= 400:
                import logging
                logging.getLogger("proxy").error(
                    "NIM %d body=%s payload_keys=%s",
                    r.status_code, r.text[:1000], list(payload.keys()),
                )
            r.raise_for_status()
            return r.json()

        try:
            return await _attempt()
        except _RetryableRateLimit as e:
            raise RateLimitError(str(e)) from e
