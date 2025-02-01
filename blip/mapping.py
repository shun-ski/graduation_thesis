#mapping.jsonを作成
import os

def generate_label_mapping(data_dir):
    label_mapping = {}
    class_names = sorted(os.listdir(data_dir))  
    for class_id, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue  

        for image_file in os.listdir(class_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                key = f"{class_name}/{image_file}"  
                label_mapping[key] = class_id  
    return label_mapping

data_dir = "foider_path" #画像フォルダパス  
label_mapping = generate_label_mapping(data_dir)
print(f"Total images: {len(label_mapping)}")
for key, class_id in list(label_mapping.items())[:10]:  # 結果の最初の10個を表示
    print(f"Image: {key}, Class ID: {class_id}")

import json
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)
print("Label mapping saved to label_mapping.json")

