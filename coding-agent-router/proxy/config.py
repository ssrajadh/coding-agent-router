from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    nvidia_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    nim_url: str = "https://integrate.api.nvidia.com/v1"
    proxy_port: int = 8000
    router_mode: str = "all_local"


settings = Settings()
