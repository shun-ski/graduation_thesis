#blip全体の画像分類
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

input_dim = 768  # BLIP特徴量の次元数
num_classes = 30  # クラス数
model = ExtendedClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("extended_model_combined_30classes.pth"))  
model.eval()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            features = blip_model.vision_model(pixel_values=pixel_values)
            cls_token = features.last_hidden_state[:, 0, :]  # [1, 768]

        with torch.no_grad():
            outputs = model(cls_token)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted_class].item()

        class_labels = {i: f"Class {i}" for i in range(num_classes)}  
        return {
            "predicted_class": class_labels[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

def predict_folder(folder_path, output_file):
    results = []
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        result = predict_image(image_file)
        result_line = f"{image_file}: Predicted class: {result.get('predicted_class', 'Error')} (Confidence: {result.get('confidence', 'N/A'):.2f})\n"
        results.append(result_line)

    with open(output_file, "w") as f:
        f.writelines(results)

    print(f"Results saved to {output_file}")

folder_path = "foider_path"  # 推論する画像フォルダ
output_file = "prediction_results.txt"  # 保存先
predict_folder(folder_path, output_file)
