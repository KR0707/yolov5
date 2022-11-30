import torch
import cv2
import numpy as np
import time
import os
os.environ['BLINKA_FT232H'] = '1' #Setting Environmental Variable
import board
import time
import digitalio
import serial

#serial setting

ser = serial.Serial('/dev/ttyACM0', 9600) # ここのポート番号を変更
ser.readline()

#GPIO Setting

solenoidforfront = digitalio.DigitalInOut(board.C1)
solenoidforback = digitalio.DigitalInOut(board.C2)
solenoidforfront.direction = digitalio.Direction.OUTPUT
solenoidforback.direction = digitalio.Direction.OUTPUT

def getModel():
    model = torch.hub.load("../yolov5",'custom', path = "/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
    return model

def getResult(imgs, model):
    result = model(imgs, size=640)
    return result

model = getModel() # モデルを呼び出す
model.conf = 0.8 #--- 検出の下限値（<1）。設定しなければすべて検出
camera = cv2.VideoCapture(2) # カメラを呼び出す

while True:
    val_arduino = ser.readline()
    val_disp = val_arduino.strip().decode('utf-8')
    if not val_disp.isdecimal():
        continue
    val_disp = int(val_disp)
    print(val_disp)
    ret, imgs = camera.read()
    imgs = cv2.resize(imgs, (640, 480))   
    cv2.imshow('color',imgs)
    cv2.waitKey(0) 
    if val_disp < 500:
        time.sleep(0.5)
        #ここで部品を押し出すアクチュエータの処理を書く
        result = getResult(imgs, model)
        print(result) # 結果を表示
        
        for *box, conf, cls in result.xyxy[0]:  # xyxy, confidence, class
            
            s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)  #--- クラス名と信頼度を文字列変数に代入
            cc = (255,255,0)
            cc2 = (128,0,0)
            cv2.rectangle(imgs,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=cc,thickness=2,lineType=cv2.LINE_AA) #--- 枠描画
            #--- 文字枠と文字列描画
            cv2.rectangle(imgs, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
            cv2.putText(imgs, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)
            # cv2.imshow("camera", imgs) # 画像を表示
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        objects = result.pandas().xyxy[0]
        if len(objects) == 0:
            print("No object")
            continue
        objects_s = objects.sort_values(by='xmin', ascending=False)


        reverse_object = objects_s.iloc[0, 5]
        time.sleep(0.5)
        if reverse_object == 1:
            solenoidforfront.value = False
            solenoidforback.value = True
            time.sleep(0.5)
            solenoidforfront.value = False
            solenoidforback.value = False
        
        elif reverse_object == 0:
            solenoidforfront.value = True
            solenoidforback.value = False
            time.sleep(0.5)
            solenoidforfront.value = False
            solenoidforback.value = False
    