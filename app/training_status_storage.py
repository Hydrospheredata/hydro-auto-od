from dataclasses import dataclass
from enum import Enum
import os
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database, Collection

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AUTO_OD_DB_NAME = os.getenv("AUTO_OD_DB_NAME", "auto_od")


class AutoODMethodStatuses(Enum):
    PENDING = 0
    STARTED = 1
    SELECTING_MODEL = 2
    SELECTING_PARAMETERS = 3
    DEPLOYING = 4
    SUCCESS = 5
    FAILED = 6
    NOT_SUPPORTED = 7


@dataclass
class TrainingStatus:
    """Class for keeping statutes of training jobs"""
    model_version_id: int
    training_data_path: str
    state: AutoODMethodStatuses
    description: Optional[str]

    def failing(self, description: str) -> None:
        self.state = AutoODMethodStatuses.FAILED
        self.description = description

    def deploying(self, description: str) -> None:
        self.state = AutoODMethodStatuses.DEPLOYING
        self.description = description

    def success(self) -> None:
        self.state = AutoODMethodStatuses.SUCCESS
        self.description = "😃"


class TrainingStatusStorage:
    """Working with database to store training statuses"""
    @staticmethod
    def __get_mongo_client():
        return MongoClient(host=MONGO_URL, port=MONGO_PORT,
                           username=MONGO_USER, password=MONGO_PASS,
                           authSource=MONGO_AUTH_DB)

    @staticmethod
    def __db() -> Database:
        return TrainingStatusStorage.__get_mongo_client()[AUTO_OD_DB_NAME]

    @staticmethod
    def __collection() -> Collection:
        return TrainingStatusStorage.__db().model_statuses

    @staticmethod
    def find_by_model_version_id(model_version_id: int) -> Optional[TrainingStatus]:
        status_document = TrainingStatusStorage.__collection().find_one({'model_version_id': model_version_id})
        if status_document is None:
            return None
        return TrainingStatus(
            status_document.get("model_version_id"),
            status_document.get("training_data_path"),
            AutoODMethodStatuses(status_document.get("state")),
            status_document.get("description")
        )

    @staticmethod
    def save_status(status: TrainingStatus):
        TrainingStatusStorage.__collection().update_one(
            {'model_version_id': status.model_version_id},
            {
                'model_version_id': status.model_version_id,
                'training_data_path': status.training_data_path,
                'status': status.state.value,
                'description': status.description
            },
            upsert=True
        )
