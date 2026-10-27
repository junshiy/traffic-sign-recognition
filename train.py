import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

# 超参数配置
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./gtsrb_data"
SAVE_MODEL_PATH = "traffic_43cls.pth"
NUM_CLASSES = 43
PRINT_STEP = 20

# 自定义数据集
class GTSRBDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.transform = transform
        if split == "train":
            csv_path = os.path.join(root, "Train.csv")
            df = pd.read_csv(csv_path)
        else:
            csv_path = os.path.join(root, "Test.csv")
            df = pd.read_csv(csv_path)

        self.image_paths = []
        self.labels = []
        for _, row in df.iterrows():
            img_full_path = os.path.join(root, row["Path"])
            self.image_paths.append(img_full_path)
            self.labels.append(row["ClassId"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# 图像预处理：增加归一化
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_set = GTSRBDataset(DATA_ROOT, split="train", transform=train_transform)
test_set = GTSRBDataset(DATA_ROOT, split="test", transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# CNN网络
class TrafficCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TrafficCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(epoch_idx):
    model.train()
    total_loss = 0.0
    total_batch = len(train_loader)
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 打印实时进度
        if (batch_idx + 1) % PRINT_STEP == 0:
            progress = (batch_idx + 1) / total_batch * 100
            print(f"【Epoch {epoch_idx+1}】进度: {progress:.1f}% | Batch:{batch_idx+1}/{total_batch} | Batch Loss:{loss.item():.4f}")

    epoch_avg_loss = total_loss / total_batch
    return epoch_avg_loss

def test_epoch():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += torch.sum(pred == labels).item()
    acc = correct / total
    return acc

if __name__ == "__main__":
    print(f"训练设备: {DEVICE}")
    print(f"数据集目录内容：{os.listdir(DATA_ROOT)}")

    loss_list = []
    acc_list = []
    for epoch in range(EPOCHS):
        loss = train_epoch(epoch)
        acc = test_epoch()
        loss_list.append(loss)
        acc_list.append(acc)
        print(f"\n==== Epoch [{epoch+1}/{EPOCHS}] 结束 ==== | 本轮平均Loss:{loss:.4f} | TestAcc:{acc:.4f}\n")

    torch.save(model, SAVE_MODEL_PATH)
    print(f"\n训练完成，模型保存至 {SAVE_MODEL_PATH}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="train loss")
    plt.title("Train Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label="test accuracy", color="orange")
    plt.title("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()