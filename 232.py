import os
os.environ['BLINKA_FT232H'] = '1' #Setting Environmental Variable

import board
import time
import digitalio


gpio = digitalio.DigitalInOut(board.C2)
gpio.direction = digitalio.Direction.OUTPUT

while True:
    gpio.value = True
    print(gpio.value)
    time.sleep(0.3)
    gpio.value = False
    print(gpio.value)
    time.sleep(0.3)