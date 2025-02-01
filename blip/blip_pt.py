#blipのPTファイル作成
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
parent_dir = "foider_path" #画像フォルダパス  
folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
all_features = {}

for folder in folders:
    folder_path = os.path.join(parent_dir, folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')

            inputs = processor(image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            with torch.no_grad():
                features = model.vision_model(pixel_values=pixel_values)

            cls_token = features.last_hidden_state[:, 0, :].detach().cpu()
            key = f"{folder}/{image_file}"
            all_features[key] = cls_token
            print(f"CLS token extracted for {key}: {cls_token.shape}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

output_file = "combined_blip_cls_tokens.pt" #保存先
if not all_features:
    raise RuntimeError("No CLS tokens were extracted. Check the input images and model processing.")
torch.save(all_features, output_file)
print(f"All CLS tokens have been saved to {output_file}")
