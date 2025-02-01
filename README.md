

# 既存モデルの中間特徴量を用いた画像分類


##  概要

物体検出モデルとキャプション生成モデルの中間特徴量を用いた画像分類

卒業論文成果物

記載者：千葉工業大学先進工学部未来ロボティクス学科藤江研究室 保科瞬

### 使用している既存モデル
中間特徴量を得る為に2つの既存モデルを使用している

#### DETR
Facebook Researchが開発した物体検出モデル
トランスフォーマエンコーダ出力の最終層から特徴量抽出

https://github.com/facebookresearch/detr

#### BLIP
Salesforceが開発したキャプション生成モデル
イメージエンコーダ出力の最終層から特徴量抽出

https://github.com/salesforce/BLIP

## レポジトリ

### LICENSE
Apache License2.0を使用している。使用しているDETRとBLIPのLICENSEは下記の通りになっている

DETR：Apache License 2.0

BLIP：BSD 3-Clause License

### .gitignore
画像分類実行時にモデルの特徴量ファイル(ptファイル)と学習済みモデルの重みの保存にpthファイルを作成している。ファイル容量が大きいため、gitignoreとした。

### object_detection
DETRの物体検出精度を評価するために既存モデルで物体検出を実行

### detr
DETRの中間特徴量を使用した画像分類

### blip
BLIPの中間特徴量を使用した画像分類

### blip+detr
DETRとBLIPを統合した特徴量で画像分類




##  インストール
```sh
git clone git@github.com:shun-ski/graduation_thesis.git
```

## 開発環境
Python：3.10.12

PyTorch：2.0.1
