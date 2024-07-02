from yesvision.vision import Vision
from yesvision.utils import get_model

def main():
    model_type = 'yolov8'  # or 'yolov5', 'yolov10', 'rcnn'
    model_size = 'm'  # RCNN does not use model sizes like YOLO
    device = 'cuda:0,1'  # Specify device, e.g., 'cpu', 'cuda:0', 'cuda:1', 'cuda:0,1' for multiple GPUs
    data_config_path = 'data/data.yaml'  # Path to the data configuration file
    epochs = 50
    imgsz = 640
    custom_model_path = 'path/to/custom/model.pth'  # Path to the custom model for testing/validation
    video_path = 'path/to/video.mp4'  # Path to the input video
    output_path = 'path/to/output.mp4'  # File to save the output video

    # Additional YOLO-specific parameters
    yolo_params = {
        'batch_size': 16,
        'optimizer': 'Adam'
    }

    model = get_model(model_type, model_size, device)
    vision = Vision(model)

    # Train the model
    vision.train(data_config_path, epochs, imgsz, **yolo_params)

    # Test the model with a custom model path
    vision.test(data_config_path, custom_model_path)

    # Validate the model with a custom model path
    vision.validate(data_config_path, custom_model_path)

    # Predict on a video
    vision.predict_video(video_path, output_path)

if __name__ == "__main__":
    main()