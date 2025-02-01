# DETRを用いた画像分類

画像分類を実行する為にはdetr_pt.pyを実行しptファイルを生成、mapping.pyを実行しjsonファイルを生成、detr_pth.pyを実行しpthファイルを作成する必要がある。

## detr_pt.py
トランスフォーマエンコーダ出力の最終層から特徴量抽出。出力された特徴量をptファイルを作成

### 実行方法

```
python3 detr_pt.py
```
実行結果
```
save to combined_features.pt
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

## detr_pth.py
作成した、ptファイル(combined_features.pt)とマッピングファイル(label_mapping.json)を使用してpthファイルを作成する。mapping.pyとdetr_pt.pyを実行しないと実行できません

### 実行方法

```
python3 detr_pth.py
```
実行結果
```
Model saved to 'extended_model_combined_30classes.pth'
```

## detr_classification.py
DETRの特徴量を使用した画像分類。detr_pth.pyを実行しないと実行できません。

### 実行方法
画像ファイルが入っているフォルダを指定
```
folder_path = "folder_path"  # 推論する画像フォルダのパス
```
実行
```
python3 detr_classification.py
```

実行結果
```
Results saved to classification_results.txt
```
結果例
```
 Predicted class: Class 25 (Confidence: 0.52)
```