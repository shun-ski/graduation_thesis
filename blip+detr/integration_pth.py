#BLIPとDETRを統合したpthファイルの作成
import torch
from torch.utils.data import DataLoader, Dataset
import json

class CustomDataset(Dataset):
    def __init__(self, detr_features_file, blip_features_file, label_mapping_file):
        self.detr_features = torch.load(detr_features_file)  # DETR特徴量
        self.blip_features = torch.load(blip_features_file)  # BLIP特徴量
        with open(label_mapping_file, "r") as f:
            self.label_mapping = json.load(f)  

        self.data = [
            (
                torch.cat((self.detr_features[key], self.blip_features[key].squeeze(0).clone()), dim=0),  
                self.label_mapping[key],
            )
            for key in self.label_mapping
            if key in self.detr_features and key in self.blip_features
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        return feature, label

detr_features_file = "combined_features.pt"  # DETR特徴量
blip_features_file = "combined_blip_cls_tokens.pt"  # BLIP特徴量
label_mapping_file = "label_mapping.json"  # ラベルマッピング

dataset = CustomDataset(detr_features_file, blip_features_file, label_mapping_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ExtendedClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ExtendedClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

input_dim = 1024  # DETR (256) + BLIP (768)
num_classes = 30  # クラス数
model = ExtendedClassifier(input_dim, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

torch.save(model.state_dict(), "extended_model_combined_30classes_1024.pth")
print("Model saved to 'extended_model_combined_30classes_1024.pth'")
