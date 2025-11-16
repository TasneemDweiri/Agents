from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class PBaseSettings(BaseSettings):
    model_config = ConfigDict(env_file = './.env', extra="allow")