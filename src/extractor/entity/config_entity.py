from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    data_dir : Path
    



# @dataclass(frozen=True)
# class DataTransformationConfig:
#     temp_labels_dir :Path
#     original_image_dir :Path
#     train_label_dir :Path
#     test_label_dir:Path
#     train_image_dir : Path
#     test_image_dir : Path
#     yolo_config_dir: Path


# @dataclass(frozen=True)
# class ModelTrainingConfig:
#     yolo_config_dir: Path
    

# # @dataclass(frozen=True)
# # class ModelPusherConfig:
# #     pickle_dir : Path
# #     s3_bucket_pickle : 