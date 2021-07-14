from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database, Collection
from hydro_auto_od.config import config


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
        return MongoClient(host=config.mongo_url, port=config.mongo_port,
                           username=config.mongo_user, password=config.mongo_pass,
                           authSource=config.mongo_auth_db)

    @staticmethod
    def __db() -> Database:
        return TrainingStatusStorage.__get_mongo_client()[config.mongo_db]

    @staticmethod
    def __collection() -> Collection:
        return TrainingStatusStorage.__db().model_statuses

    @staticmethod
    def find_by_model_version_id(model_version_id: int) -> Optional[TrainingStatus]:
        status_document = TrainingStatusStorage.__collection().find_one({'model_version_id': model_version_id})
        if status_document is None:
            return None
        return TrainingStatus(
            model_version_id=status_document.get("model_version_id"),
            training_data_path=status_document.get("training_data_path"),
            state=AutoODMethodStatuses(status_document.get("status")),
            description=status_document.get("description"),
        )

    @staticmethod
    def save_status(status: TrainingStatus):
        TrainingStatusStorage.__collection().replace_one(
            {'model_version_id': status.model_version_id},
            {
                'model_version_id': status.model_version_id,
                'training_data_path': status.training_data_path,
                'status': status.state.value,
                'description': status.description
            },
            upsert=True
        )
