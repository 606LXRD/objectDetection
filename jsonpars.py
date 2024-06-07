import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import os

num_classes = 3
state_dict = torch.load(r'C:\Users\delat\PycharmProjects\objectDetection\models\trained_model.pth')
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features, num_classes * 4)
model.load_state_dict(state_dict)
model.eval()

image_folder = r'C:\Users\delat\PycharmProjects\objectDetection\val\img'


def test_and_save_image(image_path, model):
    class_names = {0: 'WBC', 1: 'RBC', 2: 'Platelets'}
    objects = []

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    for i, box in enumerate(prediction[0]['boxes']):
        class_label = int(prediction[0]['labels'][i])
        class_name = class_names[class_label]

        points_exterior = [
            [int(box[0]), int(box[1])],
            [int(box[2]), int(box[3])]
        ]
        points_interior = []

        object_data = {
            "id": i + 1,
            "classId": class_label,
            "classTitle": class_name,
            "points": {
                "exterior": points_exterior,
                "interior": points_interior
            }
        }

        objects.append(object_data)
    prediction_output_path = os.path.splitext(image_path)[0] + '_prediction.json'
    with open(prediction_output_path, 'w') as f:
        json.dump(objects, f)


for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    if image_name.endswith('.jpeg') or image_name.endswith('.png'):
        test_and_save_image(image_path, model)
