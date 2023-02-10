

# Receives a string from Arduino using readline()
# Requires PySerial

# (c) www.xanthium.in 2021
# Rahul.S

import serial
import time
import torch
import cv2
import numpy as np
import time
import time
import serial

#serial setting

ser = serial.Serial('/dev/ttyACM0', 9600) # ここのポート番号を変更
#ser.readline()
time.sleep(2)

#GPIO Setting
"""
def getModel():
    model = torch.hub.load("../yolov5",'custom', path = "/home/kamata/Program/yolov5/models/BKweights.pt", source='local')
    return model

def getResult(imgs, model):
    result = model(imgs, size=640)
    return result

model = getModel() # モデルを呼び出す
model.conf = 0.7 #--- 検出の下限値（<1）。設定しなければすべて検出
camera = cv2.VideoCapture(2) # カメラを呼び出す
"""

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
        time.sleep(3)
        """
        ret, imgs = camera.read()
        imgs = cv2.resize(imgs, (320, 320))
        result = getResult(imgs, model)
        cv2.imshow('detection',imgs)
        cv2.waitKey(1)
        

        """
        print("YOLO Done!")
        time.sleep(10)
        ser.write(b'r')
        complete_message = ser.readline()
        print(complete_message)
    elif val_disp == 0:
        print("No Object")
        ser.write(b'r')
        continue

ser.close()