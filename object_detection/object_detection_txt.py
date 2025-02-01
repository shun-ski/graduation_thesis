#物体検出結果を画像ごとにテキスト出力
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
from collections import Counter  

model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)
image_folder_root = "folder_path"  #画像フォルダ
output_root_folder = "folder_path"  #出力先

for folder_name in os.listdir(image_folder_root):
    subfolder_path = os.path.join(image_folder_root, folder_name)
    if not os.path.isdir(subfolder_path):  
        continue

    output_folder_path = os.path.join(output_root_folder, f"{folder_name}_results")
    os.makedirs(output_folder_path, exist_ok=True)
    output_text_file = os.path.join(output_folder_path, "detection_results.txt")

    with open(output_text_file, "w") as file:
        file.write(f"Object Detection Results for Folder: {folder_name}\n")
        file.write("=" * 50 + "\n\n")

    total_label_counts = Counter()

    
    for image_name in os.listdir(subfolder_path):
        if image_name.endswith((".jpg", ".png", ".jpeg")):  
            image_path = os.path.join(subfolder_path, image_name)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=[image], return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            detected_labels = [model.config.id2label[label.item()] for label in results["labels"]]
            label_counts = Counter(detected_labels)
            total_label_counts.update(label_counts)  # 各画像の結果を累積

            with open(output_text_file, "a") as file:
                file.write(f"Image: {image_name}\n")
                file.write("Detected objects and their counts:\n")
                for label, count in label_counts.items():
                    file.write(f"  {label}: {count}\n")
                file.write("\n")

    with open(output_text_file, "a") as file:
        file.write("Total counts for all detected labels in this folder:\n")
        for label, count in sorted(total_label_counts.items(), key=lambda x: x[1], reverse=True):  # 数の多い順にソート
            file.write(f"  {label}: {count}\n")
        file.write("\n")

    print(f"Processed folder: {folder_name}")
    print(f"Results saved to: {output_text_file}")

print(f"All detection results have been saved to their respective folders under {output_root_folder}")
