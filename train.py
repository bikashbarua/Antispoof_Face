import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import AntiSpoofNet  # your model definition

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Transforms (add light augmentation)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Dataset
train_ds = ImageFolder(root='dataset/train', transform=transform_train)
val_ds = ImageFolder(root='dataset/val', transform=transform_val)

print("📂 Class-to-index mapping:", train_ds.class_to_idx)
# Expected: {'live': 0, 'spoof': 1} or vice versa

# ✅ DataLoaders
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

# ✅ Model
model = AntiSpoofNet().to(device)

# ✅ Loss & Optimizer
loss_fn = nn.BCEWithLogitsLoss()  # use with raw logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Learning Rate Scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ✅ Accuracy function
def compute_accuracy(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)

# ✅ Training loop
epochs = 7
for epoch in range(epochs):
    model.train()
    train_loss, train_acc = 0, 0
    for imgs, labels in train_dl:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # [B, 1]

        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += compute_accuracy(outputs, labels)

    scheduler.step()

    # ✅ Validation loop
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, labels)

    print(f"📊 Epoch [{epoch+1}/{epochs}] "
          f"- Train Loss: {train_loss/len(train_dl):.4f}, Acc: {train_acc/len(train_dl):.4f} "
          f"- Val Loss: {val_loss/len(val_dl):.4f}, Acc: {val_acc/len(val_dl):.4f}")

# ✅ Save the model
torch.save(model.state_dict(), "antispoof_model.pth")
print("✅ Model saved!")