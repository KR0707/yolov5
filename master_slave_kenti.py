import serial
import time
import torch
import cv2
import numpy as np
import time
import time
import serial

# serial setting

ser = serial.Serial('/dev/ttyACM0', 9600)  # ここのポート番号を変更
time.sleep(2)
object_array = []
object_array_iterator = iter(object_array)

# GPIO Setting


def getModel():
    model = torch.hub.load("../yolov5", 'custom',
                           path="/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
    return model


def getResult(imgs, model):
    result = model(imgs, size=640)
    return result


def yolo(model, camera, object_array):
    model.conf = 0.7  # --- 検出の下限値（<1）。設定しなければすべて検出
    object_array = []
    while True:
        ret, imgs = camera.read()
        imgs = cv2.resize(imgs, (320, 320))
        result = getResult(imgs, model)
        for *box, conf, cls in result.xyxy[0]:  # xyxy, confidence, class
            s = model.names[int(cls)] + ":" + '{:.1f}'.format(float(conf) * 100)  # --- クラス名と信頼度を文字列変数に代入
            cc = (255, 255, 0)
            cc2 = (128, 0, 0)
            cv2.rectangle(imgs, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color=cc, thickness=2, lineType=cv2.LINE_AA)  # --- 枠描画
            #--- 文字枠と文字列描画
            cv2.rectangle(imgs, (int(box[0]), int(box[1]) - 20), (int(box[0]) + len(s) * 10, int(box[1])), cc, -1)
            cv2.putText(imgs, s, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)
            cv2.imshow("rectangle", imgs)  # 画像を表示

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        objects = result.pandas().xyxy[0]
        if len(objects) == 0:
            print("No object")
            continue
        objects_s = objects.sort_values(by='xmin', ascending=False)
        reverse_object = objects_s.iloc[0, 6]
        object_array = object_array.append(reverse_object)
        return object_array


model = getModel()  # モデルを呼び出す
# model.conf = 0.7  # --- 検出の下限値（<1）。設定しなければすべて検出
camera = cv2.VideoCapture(2)  # カメラを呼び出す

# まず部品２個分（検知する場所からひっくり返す場所の長さ分ひっくり返す）
yolo(model, camera, object_array)
ser.write(b'r')
complete_message = ser.readline()
print(complete_message)
print(object_array)
yolo(model, camera, object_array)
ser.write(b'r')
complete_message = ser.readline()
print(complete_message)
print(object_array)


while True:
    ser.write(b's')
    val_arduino = ser.readline()
    print(val_arduino)
    val_disp = val_arduino.strip().decode('utf-8')
    if not val_disp.isdecimal():
        continue
    val_disp = int(val_disp)
    if val_disp == 1:
        print(val_disp)
        time.sleep(1)
        yolo(model, camera, object_array)
        """
        ret, imgs = camera.read()
        imgs = cv2.resize(imgs, (320, 320))
        result = getResult(imgs, model)
        for *box, conf, cls in result.xyxy[0]:  # xyxy, confidence, class
            s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)  #--- クラス名と信頼度を文字列変数に代入
            cc = (255,255,0)
            cc2 = (128,0,0)
            cv2.rectangle(imgs,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color=cc,thickness=2,lineType=cv2.LINE_AA) #--- 枠描画
            #--- 文字枠と文字列描画
            cv2.rectangle(imgs, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
            cv2.putText(imgs, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)
            cv2.imshow("rectangle", imgs) # 画像を表示        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        objects = result.pandas().xyxy[0]
       
        if len(objects) == 0:
            print("No object")
            continue
        objects_s = objects.sort_values(by='xmin', ascending=False)
        reverse_object = objects_s.iloc[0, 6]
         """
        if object_array_iterator == 2:
            ser.write(b'r')
            complete_message = ser.readline()
            print(object_array)
            next(object_array_iterator)
            print(complete_message)
            continue
        elif object_array_iterator == 3:
            ser.write(b'l')
            complete_message = ser.readline()
            print(object_array)
            next(object_array_iterator)
            print(complete_message)
            continue
        # cv2.imshow('detection',imgs)
        cv2.waitKey(1)
        print("YOLO Done!")
        time.sleep(1)

    elif val_disp == 0:
        print("No Object")
        continue

ser.close()
camera.release()
cv2.destroyAllWindows()
