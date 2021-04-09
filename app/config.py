import os
import logging


try:
    DEBUG_ENV: bool = bool(os.getenv("DEBUG", True))
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", 5000))
    CLUSTER_ENDPOINT = os.getenv("CLUSTER_ENDPOINT", "http://localhost")
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT")

    DEFAULT_RUNTIME: str = os.getenv("DEFAULT_RUNTIME", "hydrosphere/serving-runtime-python-3.6:2.1.0")
    DEFAULT_TIMEOUT: int = os.getenv("DEFAULT_TIMEOUT", 120)

    MONGO_URL = os.getenv("MONGO_URL", "localhost")
    MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
    MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    MONGO_DB = os.getenv("MONGO_DB", "auto_od")
        
except Exception as e:
    logging.error("Failed to read service configuration: (%s)", e)
    raise e 