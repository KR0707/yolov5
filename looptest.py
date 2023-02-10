import time

i = 0

while True:
    i += 1
    if i > 20:
        print("done")
        time.sleep(0.5)
        continue
    else:
        print(i)
        time.sleep(0.5)
