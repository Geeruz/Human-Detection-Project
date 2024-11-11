import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model():
    model = ssd300_vgg16(pretrained=True)
    model.eval()
    return model

def detect_humans(model, image_path, confidence_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    human_detections = [(box, score) for box, score, label in zip(boxes, scores, labels)
                        if label == 1 and score > confidence_threshold]

    return image, human_detections

def display_results(image, detections):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, score in detections:
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'Human: {score:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.axis('off')
    plt.title('Human Detection Results (SSD)')
    plt.show()

def process_image():
    model = load_model()

    image_path = input("Enter the path to the image: ")

    try:
        image, detections = detect_humans(model, image_path)
        display_results(image, detections)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

process_image()
