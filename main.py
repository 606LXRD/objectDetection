import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset_module import DatasetClass
import torch.nn.functional as F
from torchvision import transforms

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()

model = fasterrcnn_resnet50_fpn(weights=True)

num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

dataset = DatasetClass(root_dir=r'C:\Users\delat\PycharmProjects\objectDetection\train', transform=None)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

transform = transforms.Compose([transforms.ToTensor()])

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = [transform(image).to(device) for image in images]

        new_targets = []
        for target in targets:
            if 'boxes' not in target or target['boxes'].numel() == 0:
                print("No bounding boxes found in target, skipping...")
                continue

            boxes = target['boxes']
            new_boxes = []

            for box in boxes:
                x_min = box[0][0]
                y_min = box[0][1]
                x_max = box[1][0]
                y_max = box[1][1]

                if x_min >= x_max or y_min >= y_max:
                    continue

                new_boxes.append([x_min, y_min, x_max, y_max])

            if not new_boxes:
                continue

            new_target = {
                'boxes': torch.tensor(new_boxes, dtype=torch.float32).to(device),
                'labels': target['labels'].to(device)
            }
            new_targets.append(new_target)

        loss_dict = model(images, new_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")
    torch.save(model.state_dict(), 'trained_model.pth')