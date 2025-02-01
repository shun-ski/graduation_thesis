#blipのpthファイルを作成
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ExtendedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ExtendedClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

dataset = torch.load("training_data.pth")  # 特徴量とラベルリスト
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = 768  # BLIP特徴量次元数
num_classes = 30  # クラス数
model = ExtendedClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_model(model, dataloader, criterion, optimizer, num_epochs=10)
torch.save(model.state_dict(), "extended_model_combined_30classes.pth")
print("Model saved to 'extended_model_combined_30classes.pth'")
