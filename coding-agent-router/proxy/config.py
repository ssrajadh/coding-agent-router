from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    nvidia_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    nim_url: str = "https://integrate.api.nvidia.com/v1"
    proxy_port: int = 8000
    router_mode: str = "all_local"
    # Backend model names. Override for CPU smoke tests / Mac Studio swap.
    local_model: str = "qwen3-coder-16k"
    # NIM catalog: qwen3-coder-32b was retired; the current Qwen3-coder slug is
    # 480b-a35b-instruct (MoE, 35B active params).
    frontier_model: str = "qwen/qwen3-coder-480b-a35b-instruct"
    # Directory where per-trajectory JSON files are written after each request.
    # Set to "" to disable trajectory persistence.
    trajectory_dir: str = "runs/trajectories"


settings = Settings()
