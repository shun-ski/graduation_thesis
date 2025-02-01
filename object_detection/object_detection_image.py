#物体検出結果を画像ごとに画像出力
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)
image_folder_path = "folder_path"  #画像フォルダ
output_folder_path = "folder_path"  #出力先
os.makedirs(output_folder_path, exist_ok=True)

for image_name in os.listdir(image_folder_path):
    if image_name.endswith((".jpg", ".png", ".jpeg")):  
        image_path = os.path.join(image_folder_path, image_name)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", linewidth=2))
            ax.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", color="red", fontsize=12, bbox=dict(facecolor="white", alpha=0.5))

        plt.axis("off")
        output_path = os.path.join(output_folder_path, f"detected_{image_name}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close() 

        print(f"Saved detected image to {output_path}")