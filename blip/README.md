# BLIPを用いた画像分類

画像分類を実行する為にはblip_pt.pyを実行しptファイルを生成、mapping.pyを実行しjsonファイルを生成、blip_pth.pyを実行しpthファイルを作成する必要がある。

## blip_pt.py
イメージエンコーダ出力の最終層から特徴量抽出。出力された特徴量をptファイルを作成

### 実行方法  
画像ファイルが入っているフォルダを指定
```
parent_dir = "foider_path" #画像フォルダパス 
```
実行
```
python3 blip_pt.py
```
実行結果
```
All CLS tokens have been saved to combined_blip_cls_tokens.pt
```

## mapping.py
画像を分類の種類別にマッピングする。結果をjsonファイルに保存

### 実行方法
画像ファイルが入っているフォルダを指定
```
data_dir = "foider_path"  # 画像が保存されているフォルダパス
```

実行
```
python3 mapping.py
```
実行結果
```
Label mapping saved to label_mapping.json
```

## blip_pth.py
作成した、ptファイル(combined_blip_cls_tokens.pt)とマッピングファイル(label_mapping.json)を使用してpthファイルを作成する。mapping.pyとblip_pt.pyを実行しないと実行できません

### 実行方法

```
python3 blip_pth.py
```
実行結果
```
Model saved to 'extended_model_combined_30classes.pth'
```

## detr_classification.py
blipの特徴量を使用した画像分類。blip_pth.pyを実行しないと実行できません。

### 実行方法
推論するフォルダを指定
```
folder_path = "foider_path"  # 推論する画像フォルダ
```
実行
```
python3 blip_classification.py
```

実行結果
```
Results saved to prediction_results.txt
```
結果例
```
 Predicted class: Class 20 (Confidence: 0.86)

```