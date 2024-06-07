import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from PIL import Image, ImageDraw
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pandas as pd
import matplotlib.pyplot as plt

num_classes = 3
state_dict = torch.load(r'C:\Users\delat\PycharmProjects\objectDetection\models\trained_model.pth')
model = fasterrcnn_resnet50_fpn(weights=False)
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features,num_classes)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features, num_classes * 4)
model.load_state_dict(state_dict)
model.eval()

image_folder = r'C:\Users\delat\PycharmProjects\objectDetection\test\img'

def test_and_save_image(image_path, model):
    class_colors = {0: 'blue', 1: 'red', 2: 'green'}
    class_names = {0: 'WBC', 1: 'RBC', 2: 'Platelets'}

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    for i, box in enumerate(prediction[0]['boxes']):
        class_label = int(prediction[0]['labels'][i])
        class_name = class_names[class_label]
        color = class_colors[class_label]

        print(f"Object {i + 1}: Class - {class_name}, Box - {box}, Score - {prediction[0]['scores'][i]}")

        draw = ImageDraw.Draw(image)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color)
        text = f'{class_name} - {prediction[0]["scores"][i]:.3f}'
        draw.text((box[0], box[1]), text, fill=color)
    image.save(os.path.splitext(image_path)[0] + '_annotated.jpg')

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    if image_name.endswith('.jpeg') or image_name.endswith('.png'):
        test_and_save_image(image_path, model)

