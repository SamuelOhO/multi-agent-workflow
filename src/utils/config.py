"""Application configuration management.

This module provides configuration loading from environment variables and YAML files.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogFormat(str, Enum):
    """Log output format types."""

    JSON = "json"
    CONSOLE = "console"


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: str = Field(default="", description="Anthropic API key")


class UpstageConfig(BaseModel):
    """Upstage API configuration."""

    api_key: str = Field(default="", description="Upstage API key")
    base_url: str = Field(
        default="https://api.upstage.ai/v1", description="Upstage API base URL"
    )


class AppSettings(BaseModel):
    """Application settings."""

    env: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    name: str = Field(default="Agent Orchestrator", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: LogFormat = Field(default=LogFormat.JSON, description="Log format")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper


class RedisConfig(BaseModel):
    """Redis configuration (optional)."""

    url: str | None = Field(default=None, description="Redis connection URL")
    enabled: bool = Field(default=False, description="Enable Redis message bus")


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key_secret: str | None = Field(default=None, description="API key secret")
    jwt_secret: str | None = Field(default=None, description="JWT secret key")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests: int = Field(default=100, description="Max requests per period")
    period: int = Field(default=60, description="Period in seconds")


class AgentDefaults(BaseModel):
    """Default settings for agents."""

    model: str = Field(
        default="claude-haiku-4-5-20251001", description="Default LLM model"
    )
    max_tokens: int = Field(default=4096, description="Default max tokens")
    temperature: float = Field(default=0.7, description="Default temperature")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class TimeoutConfig(BaseModel):
    """Timeout configuration."""

    agent: int = Field(default=120, description="Agent timeout in seconds")
    conversation: int = Field(
        default=300, description="Conversation timeout in seconds"
    )

    @field_validator("agent", "conversation")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""

    enabled: bool = Field(default=False, description="Enable Langfuse")
    public_key: str = Field(default="", description="Langfuse public key")
    secret_key: str = Field(default="", description="Langfuse secret key")
    host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    app: AppSettings = Field(default_factory=AppSettings)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    upstage: UpstageConfig = Field(default_factory=UpstageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    agent_defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "AppConfig":
        """Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            AppConfig instance populated from environment
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            app=AppSettings(
                env=Environment(os.getenv("APP_ENV", "development")),
                debug=os.getenv("APP_DEBUG", "false").lower() == "true",
                host=os.getenv("APP_HOST", "0.0.0.0"),
                port=int(os.getenv("APP_PORT", "8000")),
            ),
            anthropic=AnthropicConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            ),
            upstage=UpstageConfig(
                api_key=os.getenv("UPSTAGE_API_KEY", ""),
                base_url=os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1"),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=LogFormat(os.getenv("LOG_FORMAT", "json")),
            ),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL"),
                enabled=os.getenv("REDIS_URL") is not None,
            ),
            security=SecurityConfig(
                api_key_secret=os.getenv("API_KEY_SECRET"),
                jwt_secret=os.getenv("JWT_SECRET"),
            ),
            rate_limit=RateLimitConfig(
                requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
                period=int(os.getenv("RATE_LIMIT_PERIOD", "60")),
            ),
            agent_defaults=AgentDefaults(
                model=os.getenv("DEFAULT_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "4096")),
                temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            ),
            timeout=TimeoutConfig(
                agent=int(os.getenv("AGENT_TIMEOUT", "120")),
                conversation=int(os.getenv("CONVERSATION_TIMEOUT", "300")),
            ),
            langfuse=LangfuseConfig(
                enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            AppConfig instance populated from YAML

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML content is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML content must be a dictionary")

        return cls._from_yaml_dict(data)

    @classmethod
    def _from_yaml_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create config from parsed YAML dictionary."""
        config_data: dict[str, Any] = {}

        # Map YAML keys to config fields
        if "app" in data:
            app_data = data["app"]
            if "env" in app_data:
                app_data["env"] = Environment(app_data["env"])
            config_data["app"] = AppSettings(**app_data)

        if "anthropic" in data:
            config_data["anthropic"] = AnthropicConfig(**data["anthropic"])

        if "upstage" in data:
            config_data["upstage"] = UpstageConfig(**data["upstage"])

        if "logging" in data:
            log_data = data["logging"]
            if "format" in log_data:
                log_data["format"] = LogFormat(log_data["format"])
            config_data["logging"] = LoggingConfig(**log_data)

        if "redis" in data:
            config_data["redis"] = RedisConfig(**data["redis"])

        if "security" in data:
            config_data["security"] = SecurityConfig(**data["security"])

        if "rate_limit" in data:
            config_data["rate_limit"] = RateLimitConfig(**data["rate_limit"])

        if "agent_defaults" in data:
            config_data["agent_defaults"] = AgentDefaults(**data["agent_defaults"])

        if "timeout" in data:
            config_data["timeout"] = TimeoutConfig(**data["timeout"])

        if "langfuse" in data:
            config_data["langfuse"] = LangfuseConfig(**data["langfuse"])

        return cls(**config_data)

    @classmethod
    def load(
        cls,
        yaml_path: str | Path | None = None,
        env_file: str | Path | None = None,
    ) -> "AppConfig":
        """Load configuration with YAML as base and environment overrides.

        Environment variables take precedence over YAML settings.

        Args:
            yaml_path: Optional path to YAML configuration file
            env_file: Optional path to .env file

        Returns:
            AppConfig instance with merged configuration
        """
        # Load from YAML if provided
        if yaml_path:
            config = cls.from_yaml(yaml_path)
        else:
            config = cls()

        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Override with environment variables if set
        return cls._apply_env_overrides(config)

    @classmethod
    def _apply_env_overrides(cls, config: "AppConfig") -> "AppConfig":
        """Apply environment variable overrides to existing config."""
        data = config.model_dump()

        # App settings
        if os.getenv("APP_ENV"):
            data["app"]["env"] = os.getenv("APP_ENV")
        if os.getenv("APP_DEBUG"):
            data["app"]["debug"] = os.getenv("APP_DEBUG", "").lower() == "true"
        if os.getenv("APP_HOST"):
            data["app"]["host"] = os.getenv("APP_HOST")
        if os.getenv("APP_PORT"):
            data["app"]["port"] = int(os.getenv("APP_PORT", "8000"))

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            data["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")

        # Upstage
        if os.getenv("UPSTAGE_API_KEY"):
            if "upstage" not in data:
                data["upstage"] = {}
            data["upstage"]["api_key"] = os.getenv("UPSTAGE_API_KEY", "")
        if os.getenv("UPSTAGE_BASE_URL"):
            if "upstage" not in data:
                data["upstage"] = {}
            data["upstage"]["base_url"] = os.getenv("UPSTAGE_BASE_URL", "")

        # Logging
        if os.getenv("LOG_LEVEL"):
            data["logging"]["level"] = os.getenv("LOG_LEVEL", "INFO")
        if os.getenv("LOG_FORMAT"):
            data["logging"]["format"] = os.getenv("LOG_FORMAT", "json")

        # Redis
        if os.getenv("REDIS_URL"):
            data["redis"]["url"] = os.getenv("REDIS_URL")
            data["redis"]["enabled"] = True

        # Security
        if os.getenv("API_KEY_SECRET"):
            data["security"]["api_key_secret"] = os.getenv("API_KEY_SECRET")
        if os.getenv("JWT_SECRET"):
            data["security"]["jwt_secret"] = os.getenv("JWT_SECRET")

        # Rate limit
        if os.getenv("RATE_LIMIT_REQUESTS"):
            data["rate_limit"]["requests"] = int(
                os.getenv("RATE_LIMIT_REQUESTS", "100")
            )
        if os.getenv("RATE_LIMIT_PERIOD"):
            data["rate_limit"]["period"] = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

        # Agent defaults
        if os.getenv("DEFAULT_MODEL"):
            data["agent_defaults"]["model"] = os.getenv("DEFAULT_MODEL", "")
        if os.getenv("DEFAULT_MAX_TOKENS"):
            data["agent_defaults"]["max_tokens"] = int(
                os.getenv("DEFAULT_MAX_TOKENS", "4096")
            )
        if os.getenv("DEFAULT_TEMPERATURE"):
            data["agent_defaults"]["temperature"] = float(
                os.getenv("DEFAULT_TEMPERATURE", "0.7")
            )

        # Timeout
        if os.getenv("AGENT_TIMEOUT"):
            data["timeout"]["agent"] = int(os.getenv("AGENT_TIMEOUT", "120"))
        if os.getenv("CONVERSATION_TIMEOUT"):
            data["timeout"]["conversation"] = int(
                os.getenv("CONVERSATION_TIMEOUT", "300")
            )

        # Langfuse
        if os.getenv("LANGFUSE_ENABLED"):
            data["langfuse"]["enabled"] = (
                os.getenv("LANGFUSE_ENABLED", "").lower() == "true"
            )
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            data["langfuse"]["public_key"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        if os.getenv("LANGFUSE_SECRET_KEY"):
            data["langfuse"]["secret_key"] = os.getenv("LANGFUSE_SECRET_KEY", "")
        if os.getenv("LANGFUSE_HOST"):
            data["langfuse"]["host"] = os.getenv("LANGFUSE_HOST", "")

        return cls._from_yaml_dict(data)


# Global configuration instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Returns:
        The global AppConfig instance

    Raises:
        RuntimeError: If configuration has not been initialized
    """
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config


def init_config(
    yaml_path: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Initialize the global configuration.

    Args:
        yaml_path: Optional path to YAML configuration file
        env_file: Optional path to .env file

    Returns:
        The initialized AppConfig instance
    """
    global _config
    _config = AppConfig.load(yaml_path=yaml_path, env_file=env_file)
    return _config


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
