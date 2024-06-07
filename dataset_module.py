import os
import json
import torch
from PIL import Image

class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(os.path.join(root_dir, 'img'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'img', self.image_paths[idx])
        json_path = os.path.join(self.root_dir, 'ann', os.path.splitext(self.image_paths[idx])[0] + '.jpeg.json')

        image = Image.open(img_path).convert("RGB")
        with open(json_path, 'r') as f:
            data = json.load(f)

        boxes = []
        labels = []

        for obj in data["objects"]:
            exterior = obj['points']['exterior']
            x_coordinates = [point[0] for point in exterior]
            y_coordinates = [point[1] for point in exterior]
            x_min = min(x_coordinates)
            y_min = min(y_coordinates)
            x_max = max(x_coordinates)
            y_max = max(y_coordinates)

            boxes.append([[x_min, y_min], [x_max, y_max]])


            label_map = {'WBC': 0, 'RBC': 1, 'Platelets': 2}
            label = obj['classTitle']
            numerical_label = label_map[label]
            labels.append(numerical_label)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }


        if self.transform:

            image = self.transform(image)

        return image, target
