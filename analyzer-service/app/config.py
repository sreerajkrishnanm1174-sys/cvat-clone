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
    annotation_refine: bool = True
    analyzer_root_path: str = ""
    analyzer_webhook_url: str = ""
    # How close a detection must be to keep an existing track id.
    track_match_minimum: float = 0.3
    # Frames an unmatched track stays alive so a lost object can be re-identified.
    track_lost_frames: int = 45
    # Center-distance gate, as a multiple of the box diagonal.
    track_gate: float = 2.0
    # At or above this, the link is a normal update. Below it, the link is kept but flagged.
    track_high_confidence: float = 0.55


def get_settings() -> Settings:
    return Settings()
