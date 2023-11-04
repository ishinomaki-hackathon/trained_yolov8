# YOLOを使用した暴力検出システム

## プロジェクトの目的
このプロジェクトの目的は、カスタムトレーニングされたYOLO（You Only Look Once）モデルを使用してリアルタイムの暴力検出システムを作成することです。このシステムは、ビデオフィードを分析し、潜在的な暴力行為を特定し、即時行動のためのアラートを発するように設計されています。

### プレビュー
![出力 gif](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/results.gif)

## 方法論

### データ準備
モデルをトレーニングするために、慎重にキュレートされたデータセットが使用されました。データセットには、「Violent」と「Non-Violent」の行動のラベル付けされたインスタンスが含まれており、バランスが取れた多様なトレーニング素材を提供するように準備されています。Roboflowデータセットには[こちら](https://universe.roboflow.com/east-west-uniersity/violance-nonviolance)からアクセスできます。

### モデルトレーニング
私たちは、リアルタイム検出能力を維持しつつ高い精度を保つことに焦点を当てた、YOLOモデルのバリアントを特定のユースケースに最適化して使用しました。

### 実装
`main.py`スクリプトは、システムのエントリーポイントとして機能し、トレーニングされたYOLOモデルを使用したリアルタイムビデオ処理と検出を実行します。

## システムのワークフロー

### ステップ1：一般的なオブジェクト検出
```python
from ultralytics import YOLO
# 標準のYOLOモデルをロードする
standard_model = YOLO('path_to_standard_weights.pt')
```

## 主要コンポーネント

### メールアラートシステム
```python
Copy code
def send_frame_as_email(frame, to_email):
    # 暴力検出時に画像フレームとともにメールアラートを送信する関数
```
システムには、ビデオフィードで暴力が検出された場合に関連するフレームが添付されてアラートを送信するメール通知機能が装備されています。<br>
**スニペット**<br><br>
![email snapshot](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/Screenshot%202023-11-04%20at%2013.32.43.png)

**送信されたフレーム:** <br><br>
![Frame sent in email](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/frame.jpg)

## はじめに（二重モデル暴力検出システム）

このシステムは、ビデオストリームで暴力を正確に検出するために、二つの異なるYOLOモデルを使用しています。最初のモデルは、フレーム内の潜在的な対象を特定する一般的なYOLO検出器です。二人以上の個人が検出された場合、二番目の専門的な事前トレーニングされたYOLOモデルが彼らの相互作用を分析して暴力の存在を決定します。

## システムのワークフロー

### ステップ1：一般的なオブジェクト検出
```python
from ultralytics import YOLO
# 標準のYOLOモデルをロードする
standard_model = YOLO('path_to_standard_weights.pt')
ビデオフィードに人が存在することを検出するのは標準のYOLOモデルの責任です。
ステップ2：暴力検出分析
```
```python
# 暴力検出用の専門的なYOLOモデルをロードする
violence_model = YOLO('path_to_violence_weights.pt')
```
標準モデルが二人以上の人を検出したとき、暴力行為のさらなる分析のために専門的な暴力検出モデルが使用されます。


### ステップ3：連続モニタリングとアラート
システムはビデオフィードを連続的に監視し、10フレーム以上の暴力が確認された場合、事件のスナップショットを含むメールアラートを引き金します。

```python
# モニタリングループの擬似コード
while True:
    # フレームごとにキャプチャ
    ret, frame = cap.read()
    # 個人の初期検出
    detections = standard_model.predict(frame)
    # 二人以上検出された場合、暴力検出モデルを使用する
    if len(detections) >= 2:
        violence_detections = violence_model.predict(frame)
        # 暴力が一貫して検出された場合、メールアラートを送信する
        if is_violence_detected(violence_detections):
            # 10回の一貫した検出後にメールを送信する
            if violence_duration > 10:
                send_frame_as_email(frame, to_email)
```
**プロンプト出力スナップショット**<br><br>
![prompt output snapshot](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/Screenshot%202023-11-04%20at%2012.58.12.png)

### リアルタイムビデオ処理
```python
cap = cv2.VideoCapture(0)
このスクリプトはライブビデオフィードをキャプチャし、訓練されたYOLOモデルを適用して暴力を検出し、必要に応じてアラートをトリガーします。
```
## 結果と成果

### データセット相関図
![Dataset Correlogram](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/labels_correlogram.jpg)
相関図は、データセット内の異なるクラスの相関関係と分布を示しており、データセットのバランスと多様性を強調しています。

### 混同行列
![Normalized Confusion Matrix](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/confusion_matrix_normalized.png)
正規化された混同行列は、モデルの分類精度を表示し、暴力行為の検出における高い真陽性率と低い偽陽性率を示しています。

### トレーニング結果
![Training Results](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/results.png)
モデルのトレーニングパフォーマンスを示すグラフであり、損失関数の最適化と、精度やリコールなどの主要指標の改善を図っています。

検証予測
![Validation preds](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/val_batch2_pred.jpg)
**予測** <br>

![Validation labels](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/val_batch2_labels.jpg)
**基本事実またはラベル（これはトレーニングの初期段階です** <br>

検証画像における予測と実際のラベルは、未知のデータに対するモデルの予測精度を視覚的に確認するためのものです。

## 用途

このシステムは、公共の場所、学校、警察機関が暴力事件を効果的に監視し、対応するために利用することができます。

## チーム協力

このシステムの開発は、データセットアノテーション、モデルトレーニング、システム統合など、さまざまな側面でのチームメンバーの協力によるものです。私たちは独自のデータセットを開発し、トレーニングすることにしましたが、アノテーションにかかる時間を節約するためにRoboflowデータセットから多くの助けを借りました。

## 使用法

暴力検出システムを実行するには：

- データセットへの正しいパスを含むdata.yamlファイルを更新してください。
- main.py内のメール設定を構成して、アラートを受け取れるようにしてください。
- 検出システムを開始するには、[main.py](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/ceab733f53d153e0e2fc24f01602d3d4243a78d9/main.py)スクリプトを実行します。
- 2つのモデルが使用されました：1つは通常のyoloによるオブジェクト検出用で、もう1つはカスタムデータセット用です。こちらから[weights](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/best.pt)をダウンロードし、[main.py](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/ceab733f53d153e0e2fc24f01602d3d4243a78d9/main.py)ファイルでロードしてください。

## セットアップと設定

### 前提条件
- Python 3.6以降
- OpenCVライブラリ
- Ultralytics YOLOライブラリ
- メール通知用のSMTPサーバーアクセス


## 結論

私たちの暴力検出システムは、人工知能の力を活用した自動監視とリアルタイムアラートを通じて、公共の安全を向上させるための重要なステップを表しています。

システムの設定と使用方法の詳細については、リポジトリに提供されている[main.py](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/ceab733f53d153e0e2fc24f01602d3d4243a78d9/main.py)スクリプトの詳細なコメントと、[Jupyter notebook](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/violence_detection_custom_dataset_yolo_train.ipynb)を参照してください。

