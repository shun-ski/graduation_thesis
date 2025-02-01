#DETR特徴量を抽出
import os
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
parent_dir = "folder_path"  # 特徴量を抽出する画像ディレクトリパス
folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
features_dict = {}

for folder in folders:
    folder_path = os.path.join(parent_dir, folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            detr_outputs = detr_model(**inputs)

        encoder_features = detr_outputs.encoder_last_hidden_state[0]  
        averaged_features = encoder_features.mean(dim=0)  
        key = f"{folder}/{image_file}"
        features_dict[key] = averaged_features
        print(f"Processed: {key}")

# 特徴量の保存
output_file = "combined_features.pt"
torch.save(features_dict, output_file)  
print(f"save to {output_file}")
