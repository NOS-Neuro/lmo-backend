"""
Configuration management using Pydantic Settings.
Validates all environment variables at startup for fail-fast behavior.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with validation."""

    # API Keys (at least one required for scanning)
    PERPLEXITY_API_KEY: Optional[str] = Field(
        default=None,
        description="Perplexity API key for web search (primary provider)"
    )

    # Optional AI settings
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for fallback scanning"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for fallback"
    )

    # Perplexity settings
    PERPLEXITY_MODEL: str = Field(
        default="sonar-pro",
        description="Perplexity model to use"
    )
    PERPLEXITY_TIMEOUT: int = Field(
        default=90,
        ge=10,
        le=300,
        description="Perplexity API timeout in seconds (10-300, default: 90)"
    )

    # Database settings
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string (optional)"
    )
    DB_POOL_MIN: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum database pool connections"
    )
    DB_POOL_MAX: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum database pool connections"
    )

    # Email settings
    RESEND_API_KEY: Optional[str] = Field(
        default=None,
        description="Resend API key for email notifications"
    )
    NOTIFY_EMAIL_FROM: Optional[str] = Field(
        default=None,
        description="From email address for notifications"
    )
    NOTIFY_EMAIL_TO: Optional[str] = Field(
        default=None,
        description="To email address for notifications"
    )

    # Frontend settings
    FRONTEND_ORIGIN: str = Field(
        default="http://localhost:3000",
        description="Frontend origin for CORS (no wildcard allowed)"
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Performance settings
    MAX_COMPETITOR_SCAN_WORKERS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum parallel workers for competitor scans (1-10)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v_upper

    @field_validator("DB_POOL_MAX")
    @classmethod
    def validate_pool_max_gte_min(cls, v: int, info) -> int:
        """Ensure max pool size >= min pool size."""
        if "DB_POOL_MIN" in info.data and v < info.data["DB_POOL_MIN"]:
            raise ValueError("DB_POOL_MAX must be >= DB_POOL_MIN")
        return v

    @field_validator("FRONTEND_ORIGIN")
    @classmethod
    def validate_no_wildcard_origin(cls, v: str) -> str:
        """Prevent wildcard CORS origin."""
        if v == "*":
            raise ValueError(
                "FRONTEND_ORIGIN cannot be '*' (wildcard). "
                "Set explicit origin or leave unset for localhost:3000 default."
            )
        return v

    @property
    def email_notifications_enabled(self) -> bool:
        """Check if email notifications are fully configured."""
        return bool(
            self.RESEND_API_KEY
            and self.NOTIFY_EMAIL_FROM
            and self.NOTIFY_EMAIL_TO
        )

    @property
    def database_enabled(self) -> bool:
        """Check if database is configured."""
        return bool(self.DATABASE_URL)


# Global settings instance
settings = Settings()
