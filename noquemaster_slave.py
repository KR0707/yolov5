import queue
import time

import cv2
import numpy as np
import serial
import torch

# serial setting

ser = serial.Serial('/dev/ttyACM_arduino_mega2560', 9600)  # ここのポート番号を変更
time.sleep(2)
object_array = []
in_the_box = 2
next_part = 2
past_part = 2
#object_array_queue = queue.Queue()

# GPIO Setting


def getModel():
    model = torch.hub.load("../yolov5", 'custom',
                           path="/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
    return model


def getResult(img, model):
    result = model(img, size=640)
    return result


def yolo(model, camera, object_array):
    model.conf = 0.7  # --- 検出の下限値（<1）。設定しなければすべて検出
    while True:
        ret, imgs = camera.read()
        gamma     = 0.8                              # γ値を指定
        img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値

# 公式適用
        for i in range(256):
            img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)

# 読込画像をガンマ変換
        imgs = cv2.convertScaleAbs(imgs, alpha=1.3, beta=20)
        gamma_img = cv2.LUT(imgs,img2gamma)
        #imgs = cv2.resize(imgs, (320, 320))
        result = getResult(gamma_img, model)
        for *box, conf, cls in result.xyxy[0]:  # xyxy, confidence, class
            s = model.names[int(cls)] + ":" + '{:.1f}'.format(float(conf) * 100)  # --- クラス名と信頼度を文字列変数に代入
            cc = (255, 255, 0)
            cc2 = (128, 0, 0)
            cv2.rectangle(imgs, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color=cc, thickness=2, lineType=cv2.LINE_AA)  # --- 枠描画
            #--- 文字枠と文字列描画
            cv2.rectangle(imgs, (int(box[0]), int(box[1]) - 20), (int(box[0]) + len(s) * 10, int(box[1])), cc, -1)
            cv2.putText(imgs, s, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)
        cv2.imshow("rectangle", imgs)  # 画像を表示
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        objects = result.pandas().xyxy[0]
        if len(objects) == 0:
            #print("No object by YOLO")
            time.sleep(0.2)
            continue
        else:
            #print("Object detected")
            break

    objects_s = objects.sort_values(by='xmin')
    reverse_object = objects_s.iloc[0, 5]
    
    # object_array.append(reverse_object)
    return reverse_object


model = getModel()  # モデルを呼び出す
model.conf = 0.5  # --- 検出の下限値（<1）。設定しなければすべて検出
camera = cv2.VideoCapture(2)  # カメラを呼び出す

""""
yolo(model, camera, object_array)
time.sleep(2)
object_array_queue.put(object_array)
ser.write(b'r')
complete_message = ser.readline()
#print(complete_message)
#print(object_array)
time.sleep(2)

yolo(model, camera, object_array)
time.sleep(2)
object_array_queue.put(object_array)
ser.write(b'r')
complete_message = ser.readline()
#print(complete_message)
#print(object_array)
time.sleep(2)
"""

while True:
    gamma     = 0.8                             # γ値を指定
    img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値
    
# 公式適用
    for i in range(256):
        img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)

# 読込画像をガンマ変換

    ret, imgs = camera.read()
    imgs = cv2.convertScaleAbs(imgs, alpha=1.3, beta=20)
    gamma_img = cv2.LUT(imgs,img2gamma)
    results = model(gamma_img, size=640)
    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class

      #--- クラス名と信頼度を文字列変数に代入
        s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)

      #--- ヒットしたかどうかで枠色（cc）と文字色（cc2）の指定
        cc = (0,255,255)
        cc2 = (0,128,128)

      #--- 枠描画
        cv2.rectangle(imgs,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=cc,thickness=2,)
      #--- 文字枠と文字列描画
        cv2.rectangle(imgs, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
        cv2.putText(imgs, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)

  #--- 描画した画像を表示
    cv2.imshow('color',imgs)
  
  #--- 「q」キー操作があればwhileループを抜ける ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ser.write(b's')
    time.sleep(2)
    val_arduino = ser.readline()
    #print("sensor value: " + val_arduino.decode('utf-8'))
    val_disp = val_arduino.strip().decode('utf-8')
    if not val_disp.isdecimal():
        continue
    val_disp = int(val_disp)
    #print(val_disp)
    if val_disp == 1:
        time.sleep(2)
        if in_the_box == 2:
            ser.write(b'l')
            ser.readline()
            time.sleep(0.5)
            #next_part = 0
            next_part = yolo(model, camera, object_array)
            print("left")
            complete_message = ser.readline()
            in_the_box = next_part
            print("in_the_box: BKF, next_part: "+ str(next_part))
            continue
        elif in_the_box  == 3:
            ser.write(b'r')
            ser.readline()
            time.sleep(0.5)
            #next_part = 0
            next_part = yolo(model, camera, object_array)
            print("right")
            complete_message = ser.readline()
            in_the_box = next_part
            print("in_the_box: BKB, next_part: " + str(next_part))
            continue
        # cv2.imshow('detection',imgs)
        #cv2.waitKey(1)
        #print("YOLO Done!")
        time.sleep(2)

    elif val_disp == 0:
        #print("No Object by sensor")
        continue

ser.close()
camera.release()
cv2.destroyAllWindows()
