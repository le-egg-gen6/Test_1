import os
import yaml
import torch
from ultralytics import YOLO

def create_yolo_dataset_yaml_configuration(dataset_path, output_yaml_file, output_class_file, train_dir="Training", test_dir="Test"):
    """
    Creates a YAML configuration file for YOLOv8 based on a dataset with class directories.
    
    Args:
        dataset_path (str): Path to the root of your dataset
        output_yaml_path (str): Path where to save the YAML file
        train_dir (str): Name of the training directory (default: 'Training')
        test_dir (str): Name of the testing/validation directory (default: 'Test')
    """
    train_path = os.path.join(dataset_path, train_dir)
    
    class_names = []
    diff_class_names = []

    if os.path.exists:
        for d in os.listdir(train_path):
            class_names.append(d)
            splited_name = d.split(" ")
            base_class_name = splited_name[0]
            if base_class_name not in diff_class_names:
                diff_class_names.append(base_class_name)

    class_names.sort()
    diff_class_names.sort()

    yaml_content = {
        'path': os.path.abspath(dataset_path),
        'train': train_dir,
        'val': test_dir,
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    base_fruits_file = os.path.join(os.path.dirname(output_yaml_file), output_class_file)
    with open(base_fruits_file, 'w') as f:
        for fruit in diff_class_names:
            f.write(f"{fruit}\n")
    
    print(f"Created YAML config at {output_yaml_file}")
    print(f"Created unique fruit names file at {base_fruits_file}")
    print(f"Detected {len(class_names)} full classes")
    print(f"Detected {len(diff_class_names)} unique base fruit types")


def pretrain_model(base_model_name, dataset_config_path, epochs, imagesz, batch, model_name):
    model = YOLO(base_model_name)

    model.train(
        data=dataset_config_path,
        epochs=epochs,
        imgsz=imagesz,
        batch=batch,
        name=model_name
    )

YOLOv8_BASE_MODEL = "yolov8n.pt"
DATASET_TRAIN_CONFIG = "train_dataset.yaml"
UNIQUE_CLASS_CONFIG = "unique_fruit_name.yaml"
EPOCHS=50
IMAGE_SIZE=100
BATCH=16
MODEL_NAME = "fruit_detect_360"

DATASET_PATH = "fruits-360_100x100/fruits-360"

def main():

    if not os.path.isfile(DATASET_TRAIN_CONFIG):
        create_yolo_dataset_yaml_configuration(
            dataset_path=DATASET_PATH,
            output_yaml_file=DATASET_TRAIN_CONFIG,
            output_class_file=UNIQUE_CLASS_CONFIG,
            train_dir="Training",
            test_dir="Test"
        )
    
    pretrain_model(
        base_model_name=YOLOv8_BASE_MODEL,
        dataset_config_path=DATASET_TRAIN_CONFIG,
        epochs=EPOCHS,
        imagesz=IMAGE_SIZE,
        batch=BATCH,
        model_name=MODEL_NAME
    )
    # print(torch.cuda.is_available()) # False


if __name__ == "__main__":
    main()