"""
Main settings of the app.
"""

from pydantic import Field

from .base import PBaseSettings


class AllSettings(PBaseSettings):
    """..."""

    D_HOST: str | None = Field(default=None)
    D_PORT: int | None = Field(default=None)
    D_USER: str | None = Field(default=None)
    D_PASSWORD: str | None = Field(default=None)
    OPENAI_API_KEY: str | None = Field(default=None)
    OPENAI_BASE_URL: str | None = Field(default=None)
    MISTRAL_API_KEY: str | None = Field(default=None)
