import os
import random
import cv2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from ..base_model import DetectionModel
from ..utils import read_data_config

class RCNNModel(DetectionModel):
    def __init__(self, model_config='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', model_weights='detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl', device='cpu'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_config)
        self.cfg.MODEL.WEIGHTS = model_weights
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.predictor = DefaultPredictor(self.cfg)

    def train(self, data_path, epochs, imgsz, **kwargs):
        data = read_data_config(data_path)
        self.cfg.DATASETS.TRAIN = (data["train"],)
        self.cfg.DATASETS.TEST = (data["val"],)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = epochs
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = data["num_classes"]
        self.cfg.INPUT.MIN_SIZE_TRAIN = (imgsz,)
        self.cfg.INPUT.MIN_SIZE_TEST = (imgsz,)

        for key, value in kwargs.items():
            setattr(self.cfg, key, value)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def test(self, data_path, model_path, **kwargs):
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        predictor = DefaultPredictor(self.cfg)

        data = read_data_config(data_path)
        dataset_dicts = DatasetCatalog.get(data["val"])
        for d in random.sample(dataset_dicts, 3):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(data["val"]), scale=0.8)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(out.get_image()[:, :, ::-1])

    def validate(self, data_path, model_path, **kwargs):
        self.test(data_path, model_path, **kwargs)

    def predict_video(self, video_path, output_path, **kwargs):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            outputs = self.predictor(frame)
            v = Visualizer(frame[:, :, ::-1], scale=1.2)
            out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            out.write(out_frame)

        cap.release()
        out.release()
