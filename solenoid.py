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

while True:
        solenoidforfront.value = True
        solenoidforback.value = False