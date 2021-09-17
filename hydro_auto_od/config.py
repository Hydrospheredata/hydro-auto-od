from typing import Optional
from pydantic import BaseSettings

class Config(BaseSettings):
    mongo_user: str
    mongo_pass: str
    s3_endpoint: Optional[str]
    debug_env: bool = True
    grpc_port: int = 5000
    cluster_endpoint: str = "http://localhost"
    default_runtime: str = "hydrosphere/serving-runtime-python-3.7:3.0.0-dev4"
    default_timeout: int = 120
    mongo_url: str = "localhost"
    mongo_port: int = 27017
    mongo_auth_db: str = "admin"
    mongo_db: str = "auto_od"

    class Config:
        case_sensitive = False

config = Config()