import sys
import cv2
import torch

# --- 検出する際のモデルを読込 ---
# model = torch.hub.load('ultralytics/yolov5','yolov5s')#--- webのyolov5sを使用
model = torch.hub.load("../yolov5", 'custom',
                       path="/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
# --- 検出の設定 ---
model.conf = 0.5 # --- 検出の下限値（<1）。設定しなければすべて検出
# model.classes = [0] #--- 0:person クラスだけ検出する。設定しなければすべて検出
# print(model.names) #--- （参考）クラスの一覧をコンソールに表示

# --- 映像の読込元指定 ---
# camera = cv2.VideoCapture("../pytorch_yolov3/data/sample.avi")#--- localの動画ファイルを指定
camera = cv2.VideoCapture(2)  # --- カメラ：Ch.(ここでは0)を指定


while True:
    ret, imgs = camera.read()  # --- 映像から１フレームを画像として取得
    imgs = cv2.convertScaleAbs(imgs, alpha=1.0, beta=0)
# --- 推定の検出結果を取得 ---
    #results = model(imgs)  # --- サイズを指定しない場合は640ピクセルの画像にして処理
    results = model(imgs, size=320) #--- サイズを指定する場合はこちら

    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class

      #--- クラス名と信頼度を文字列変数に代入
        s = model.names[int(cls)] + ":" + '{:.1f}'.format(float(conf) * 100)
        cc = (255, 255, 0)
        cc2 = (0, 0, 0)

      #--- 枠描画
        cv2.rectangle(imgs,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=cc,thickness=2,)
      #--- 文字枠と文字列描画
        cv2.rectangle(imgs, (int(box[0]), int(box[1]) - 20), (int(box[0]) + len(s) * 10, int(box[1])), cc, -1)
        cv2.putText(imgs, s, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)


  #--- 描画した画像を表示
    cv2.imshow('color', imgs)

  # --- 「q」キー操作があればwhileループを抜ける ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
