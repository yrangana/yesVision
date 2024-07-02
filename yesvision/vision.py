from .base_model import DetectionModel

class Vision:
    def __init__(self, model: DetectionModel):
        self.model = model
        
    def set_model(self, model: DetectionModel):
        self._model = model

    def train(self, data_path, epochs, imgsz, **kwargs):
        self._model.train(data_path, epochs, imgsz, **kwargs)

    def test(self, data_path, model_path, **kwargs):
        self._model.test(data_path, model_path, **kwargs)

    def validate(self, data_path, model_path, **kwargs):
        self._model.validate(data_path, model_path, **kwargs)

    def predict_video(self, video_path, output_path, **kwargs):
        self._model.predict_video(video_path, output_path, **kwargs)