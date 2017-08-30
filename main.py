import cv2
import numpy
from PIL import Image,ImageDraw

def detectFaces(img):
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    faces = detectFaces(frame)
    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    if faces:
        for (x1,y1,x2,y2) in faces:
            draw_instance = ImageDraw.Draw(pil_im)
            draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
    open_cv_image = numpy.array(pil_im) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    cv2.imshow("capture", open_cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
