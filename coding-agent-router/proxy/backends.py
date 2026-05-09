import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaBackend:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
    async def chat_completion(self, body: dict) -> dict:
        payload = {**body, "model": self.model, "stream": False}
        r = await self.client.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        r.raise_for_status()
        return r.json()


class NIMBackend:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(timeout=300.0, headers=self.headers)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    async def chat_completion(self, body: dict) -> dict:
        payload = {**body, "model": self.model, "stream": False}
        r = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        if r.status_code == 429:
            raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
        r.raise_for_status()
        return r.json()
