from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    cvat_base_url: str = "http://localhost:8080"
    cvat_access_token: str = ""
    cvat_auth_scheme: str = "Bearer"
    webhook_secret: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    analyzer_allow_external_default: bool = False
    nuclio_invoke_url: str = "http://nuclio:8080"
    yolo_function_name: str = "onnx-wongkinyiu-yolov7"
    database_url: str = "sqlite:///./data/analyzer.sqlite3"
    redis_url: str = "redis://localhost:6379/0"
    analyzer_data_dir: str = "./data"
    analyzer_root_path: str = ""
    analyzer_webhook_url: str = ""


def get_settings() -> Settings:
    return Settings()
