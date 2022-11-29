import os
os.environ['BLINKA_FT232H'] = '1' #Setting Environmental Variable

import board
import time
import digitalio

#GPIO Setting : C0 will be output port.
sensor = digitalio.DigitalInOut(board.C0)
solenoidforfront = digitalio.DigitalInOut(board.C1)
solenoidforback = digitalio.DigitalInOut(board.C2)
sensor.direction = digitalio.Direction.INPUT
solenoidforfront.direction = digitalio.Direction.OUTPUT
solenoidforback.direction = digitalio.Direction.OUTPUT


while True:
    led.value = True
    time.sleep(0.5)
    led.value = False
    time.sleep(0.5)