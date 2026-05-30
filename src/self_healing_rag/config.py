from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vector_store_path: str = "data/vector_store"
    max_attempts: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

