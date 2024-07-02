from ultralytics import YOLO
from ..base_model import DetectionModel

class YOLOModel(DetectionModel):
    def __init__(self, version='v5', model_size='s', device='cpu'):
        self.model = YOLO(f'yolo{version}{model_size}.pt')
        self.device = device

    def train(self, data_path, epochs, imgsz, **kwargs):
        self.model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=self.device, **kwargs)

    def test(self, data_path, model_path, **kwargs):
        model = YOLO(model_path)
        results = model.val(data=data_path, split='test', device=self.device, **kwargs)
        print(results)

    def validate(self, data_path, model_path, **kwargs):
        model = YOLO(model_path)
        results = model.val(data=data_path, split='val', device=self.device, **kwargs)
        print(results)

    def predict_video(self, video_path, output_path, **kwargs):
        self.model.predict(source=video_path, save=True, project=output_path, device=self.device, **kwargs)