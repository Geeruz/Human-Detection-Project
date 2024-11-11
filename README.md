
# Human Detection Model Comparison

A comprehensive comparison of three popular object detection models (YOLO, Fast R-CNN, and SSD) for human detection tasks. This project implements and evaluates these models using various performance metrics to provide insights into their strengths and weaknesses.

##  Overview

This project compares three state-of-the-art object detection models specifically for human detection:
- YOLO (You Only Look Once)
- Fast R-CNN (Region-based Convolutional Neural Network)
- SSD (Single Shot Detector)

The comparison is based on several key metrics:
- Accuracy (mAP, Precision, Recall)
- Speed (FPS, inference time)
- Resource utilization (CPU/GPU usage, memory consumption)

##  Features

- Automated performance measurement and comparison
- COCO dataset integration for standardized testing
- GPU support for accelerated processing
- Detailed metric visualization and reporting
- Easy-to-use interface for model evaluation

##  Results

The models are evaluated on the COCO dataset (filtered for human detection) with the following metrics:
- Mean Average Precision (mAP)
- Frames Per Second (FPS)
- Resource Usage
- Model Size
- Inference Time

(Note: Add your specific results and comparisons here)

##  Requirements

### Hardware Requirements
- NVIDIA GPU (recommended)
- Minimum 8GB RAM
- 20GB free disk space

### Software Requirements
```
python >= 3.8
torch >= 1.9.0
torchvision >= 0.10.0
ultralytics >= 8.0.0
pycocotools
psutil
numpy
tqdm
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/geezuz/Human-Detection-Project.git
cd human-detection-comparison
```

2. Install required packages:
```bash
pip install -r requirements.txt
```


## 📊 Sample Output

The script generates a comprehensive comparison report including:
- Performance metrics for each model
- Resource utilization graphs
- Detection accuracy comparisons
- Processing speed analysis


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- COCO Dataset for providing the training and validation data
- Ultralytics for YOLO implementation
- Torchvision for Fast R-CNN implementation
- NVIDIA for SSD implementation



