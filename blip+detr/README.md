# BLIPとDETRを統合した特徴量で画像分類

２つのモデルの特徴量を合わせて画像分類を実行する。combined_blip_cls_tokens.pt、combined_features.pt、label_mapping.jsonはBLIPとDETRのフォルダで作成されたものを使用する。

## integration_pth.py

DETRとBLIPのptファイルを統合してpthファイルを作成する。

### 実行方法

```
python3 integration_pth.py
```
実行結果
```
Model saved to 'extended_model_combined_30classes_1024.pth'
```

## integration_classification.py
BLIPとDETRを統合した特徴量で画像分類。integration_pth.pyを実行しないと実行できません。

### 実行方法
推論するフォルダを指定
```
folder_path = "foider_path"  # 推論する画像フォルダ
```
実行
```
python3 integration_classification.py
```

実行結果
```
Results saved to classification_resultsadd.txt
```
結果例
```
 Predicted class: Class 21 (Confidence: 0.99)
```

平均正解率でBLIP単体の特徴量とDETR単体の特徴量で画像分類した時よりも上回る結果となった。