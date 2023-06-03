import cv2

#read video
#the higher resolution slower the processes
cap = cv2.VideoCapture("video/01.mp4")

minWidth = 100
minHeight = 100

#Object dedection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()

    if not ret:
        print("! Camera cannot be read !")
        break
    
    frameCopy = frame
    height,width,_=frameCopy.shape

    #crop frame
    roi = frameCopy[int(height/10)*7:int(height/10)*9, int(width/10)*1:int(width/10)*9]

    #preProcessing frame
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),5)
    #applying on each frame
    sub = object_detector.apply(blur)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(sub,cv2.MORPH_CLOSE,kernal)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernal)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frameCopy,(int(width/10)*1,int(height/10)*8),( int(width/10)*9,int(height/10)*8),(0,0,255),3)

    for (i,c) in enumerate(counterShape):
        x,y,w,h = cv2.boundingRect(c)
        validate_counter = w >= minWidth and h >= minHeight
        if validate_counter:
            cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
    #show frame
    cv2.imshow("Frame",frameCopy)

    key = cv2.waitKey(30)

    #It turns off when you press the q key.
    if key & 0xFF == ord("q"):
        print("logged out")
        break

cap.relase()
cv2.destroyAllWindows()
