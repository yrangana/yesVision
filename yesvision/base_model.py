from abc import ABC, abstractmethod

class DetectionModel(ABC):
    @abstractmethod
    def train(self, data_path, epochs, imgsz, **kwargs):
        pass

    @abstractmethod
    def test(self, data_path, model_path, **kwargs):
        pass

    @abstractmethod
    def validate(self, data_path, model_path, **kwargs):
        pass

    @abstractmethod
    def predict_video(self, video_path, output_path, **kwargs):
        pass