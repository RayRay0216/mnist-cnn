from mnist_model import SimpleCNN
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1) 裝置選擇
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) 超參數
batch_size, lr, epochs = 64, 1e-2, 1  # 先跑 1 epoch 確認流程

# 3) 前處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 常用均值/方差
])

# 4) 資料集與載入器（第一次會自動下載）
train_ds = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
test_ds  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# 6) 訓練
for epoch in range(1, epochs+1):
    model.train()
    running_loss = correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch}: train loss={running_loss/total:.4f}, acc={correct/total:.4f}")

# 7) 測試
model.eval()
t_loss = t_correct = t_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        t_loss += criterion(logits, labels).item() * images.size(0)
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total += labels.size(0)
print(f"Test: loss={t_loss/t_total:.4f}, acc={t_correct/t_total:.4f}")

# 8) 存權重
torch.save(model.state_dict(), "mnist_cnn.pt")
print("Saved weights to mnist_cnn.pt")
