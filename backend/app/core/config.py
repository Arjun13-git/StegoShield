from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_version: str = "0.1.0"
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    min_image_side: int = int(os.getenv("MIN_IMAGE_SIDE", 64))
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", 4096))
    model_path: str = os.getenv("MODEL_PATH", "data/models/stegoshield_rf.joblib")
    cors_origins: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cors_origins is None:
            object.__setattr__(self, "cors_origins", ["http://localhost:5173"])


settings = Settings()
