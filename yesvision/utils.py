import yaml
from .models.rcnn_model import RCNNModel
from .models.yolo_model import YOLOModel

def read_data_config(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_model(model_type, model_size=None, device='cpu'):
    if model_type.startswith('yolo'):
        version = model_type.replace('yolo', '')
        return YOLOModel(version, model_size, device)
    elif model_type == 'rcnn':
        return RCNNModel(device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")