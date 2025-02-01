#DETRのみのpthファイルを作成
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json

class CustomDataset(Dataset):
    def __init__(self, features_file, label_mapping_file):
        self.features = torch.load(features_file)  
        with open(label_mapping_file, "r") as f:
            self.label_mapping = json.load(f)  
        self.data = [
            (self.features[key], self.label_mapping[key])
            for key in self.label_mapping if key in self.features
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        return feature, label

features_file = "combined_features.pt"  # 特徴量ファイル
label_mapping_file = "label_mapping.json"  # ラベルマッピングファイル
dataset = CustomDataset(features_file, label_mapping_file)

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check 'combined_features.pt' and 'label_mapping.json'.")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

input_dim = 256  # DETR特徴量の次元数
num_classes = 30  # クラス数
model = ExtendedClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features, labels = features.float(), labels.long()
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "extended_model_combined_30classes.pth")
print("Model saved to 'extended_model_combined_30classes.pth'")
