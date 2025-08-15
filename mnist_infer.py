# mnist_infer.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 建立模型並載入權重
model = SimpleCNN().to(device)
state = torch.load("mnist_cnn.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# 2) 前處理跟訓練時一致
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3) 準備測試資料，隨機看 5 筆
test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
loader = DataLoader(test_ds, batch_size=1, shuffle=True)

with torch.no_grad():
    correct = 0
    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)
        pred = model(img).argmax(1).item()
        print(f"[{i+1}] 預測: {pred}  實際: {label.item()}")
        correct += int(pred == label.item())
        if i >= 4: break
    print(f"前 5 筆正確數：{correct}/5")
