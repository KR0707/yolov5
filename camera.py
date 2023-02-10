import cv2

# Video
frameWidth = 640
frameHeight = 480

#Video Source
#cap = cv2.VideoCapture('videos/traffic.mp4') #自分のmp4のpathを入力
cap = cv2.VideoCapture(2)



while True:
    ret, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    alpha = 1.0 # コントラスト項目
    beta = 60    # 明るさ項目

    # 明るさ・コントラスト操作
    img = cv2.convertScaleAbs(img,alpha = alpha,beta = beta)
    cv2.imshow('Video', img)
    print('ret=', ret)

    # qを押すと止まる。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
