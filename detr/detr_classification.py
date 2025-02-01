#detrで画像分類
import os
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

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

input_dim = 256  # DETR特徴量の次元数
num_classes = 30  # クラス数
model = ExtendedClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("extended_model_combined_30classes.pth"))
model.eval()

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

def predict_folder(folder_path, output_file):
    results = []
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for image_file in image_files:
        try:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                detr_outputs = detr_model(**inputs)
                encoder_features = detr_outputs.encoder_last_hidden_state[0]  
                averaged_features = encoder_features.mean(dim=0).unsqueeze(0)  

            with torch.no_grad():
                outputs = model(averaged_features)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted_class].item()

            class_labels = {i: f"Class {i}" for i in range(num_classes)}  
            results.append(f"{image_file}: Predicted class: {class_labels[predicted_class]} (Confidence: {confidence:.2f})\n")

        except Exception as e:
            results.append(f"{image_file}: Error - {str(e)}\n")

    with open(output_file, "w") as f:
        f.writelines(results)

    print(f"Results saved to {output_file}")

folder_path = "folder_path"  # 推論する画像フォルダのパス
output_file = "classification_results.txt"  # 結果保存先
predict_folder(folder_path, output_file)