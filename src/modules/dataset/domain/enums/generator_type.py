from enum import Enum


class DatasetGeneratorRegistry(str, Enum):
    COCO_PYMOO = "coco_pymoo"
