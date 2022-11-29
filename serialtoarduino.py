import torch
import cv2
import serial
import PIL
import numpy as np
import time
import os
os.environ['BLINKA_FT232H'] = '1' #Setting Environmental Variable
import board
import time
import digitalio


def getModel():
    model = torch.hub.load("../yolov5",'custom', path = "/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
    return model

def main():
    model = getModel() # モデルを呼び出す
    model.conf = 0.8 #--- 検出の下限値（<1）。設定しなければすべて検出
    camera = cv2.VideoCapture(0) # カメラを呼び出す

    while True:
        ret, imgs = camera.read() # カメラから画像を取得
        result = model(imgs) # 画像を読み込む
        result = model(imgs, size=640)
        df = result.pandas().xyxy[0] # データフレームに変換
        df = df[df['confidence'] > 0.8] # 信頼度が0.8以上のものを抽出
        #pos_x = 240
        print(df) # 結果を表示
        for *box, conf, cls in result.xyxy[0]:  # xyxy, confidence, class
            s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)  #--- クラス名と信頼度を文字列変数に代入
            cc = (255,255,0)
            cc2 = (128,0,0)
            """
            #--- ヒットしたかどうかで枠色（cc）と文字色（cc2）の指定
            if int(box[0])>pos_x :
                cc = (255,255,0)
                cc2 = (128,0,0)
            else:
                cc = (0,255,255)
                cc2 = (0,128,128)
            """
            cv2.rectangle(imgs,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=cc,thickness=2,lineType=cv2.LINE_AA) #--- 枠描画
            #--- 文字枠と文字列描画
            cv2.rectangle(imgs, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
            cv2.putText(imgs, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)

        cv2.imshow("camera", imgs) # 画像を表示
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()