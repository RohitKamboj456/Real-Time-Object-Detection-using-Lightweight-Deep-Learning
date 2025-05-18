import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import pickle
import argparse
 
# Define paths (adjust if needed)
TRAIN_ROOT = r"D:\New folder (2)\train"
TRAIN_CSV = r"D:\New folder (2)\train\_annotations.csv"
TEST_ROOT = r"D:\New folder (2)\test"
TEST_CSV = r"D:\New folder (2)\test\_annotations.csv"

class DetectionDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading images and annotations."""
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        # Filter only existing images
        self.imgs = [img for img in self.df['filename'].unique() if os.path.exists(os.path.join(self.root, img))]
        # Group annotations by filename
        self.annotations = self.df.groupby('filename')[['xmin', 'ymin', 'xmax', 'ymax', 'class']].apply(lambda x: x.values.tolist()).to_dict()
        # Create label map (starting from 1, background is 0 implicitly)
        self.class_labels = sorted(self.df['class'].unique())
        self.label_map = {label: idx + 1 for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return empty data if image fails to load
            return torch.zeros(3, 224, 224), {'boxes': torch.zeros((0, 4), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.int64)}

        # Get annotations
        annotations = self.annotations.get(img_name, [])
        boxes = []
        labels = []
        for ann in annotations:
            xmin, ymin, xmax, ymax, class_label = ann
            if xmin < xmax and ymin < ymax:  # Validate box
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.label_map[class_label])
            else:
                print(f"Invalid bounding box in {img_name}: {ann}")

        # Handle case with no annotations
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img = self.transform(img)

        return img, target

def collate_fn(batch):
    """Custom collate function for variable-sized targets."""
    return tuple(zip(*batch))

def train_model():
    """Main training function."""
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = DetectionDataset(TRAIN_ROOT, TRAIN_CSV, transform=transform)
    test_dataset = DetectionDataset(TEST_ROOT, TEST_CSV, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Save label map
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(train_dataset.label_map, f)

    # Model setup
    num_classes = len(train_dataset.class_labels) + 1  # Include background
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            try:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                train_loss += losses.item()
                print(f"Epoch {epoch+1}, Iteration {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            except Exception as e:
                print(f"Error in training iteration {i}: {e}")
                continue

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in test_loader:
                try:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1}, Avg Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model.pth')
            print("Saved best model")

        lr_scheduler.step()

if __name__ == "__main__":
    train_model() 