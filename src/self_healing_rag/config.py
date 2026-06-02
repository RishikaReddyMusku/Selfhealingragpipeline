from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "ollama"
    chat_model: str = "gpt-4o-mini"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"
    retrieval_backend: str = "local"
    vector_store_path: str = "data/vector_store"
    vector_collection_name: str = "self_healing_rag"
    local_index_path: str = "data/index/chunks.jsonl"
    max_attempts: int = 3
    min_context_score: float = 0.05
    min_context_term_overlap: float = 0.15
    use_llm: bool = False
    llm_fallback_enabled: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
