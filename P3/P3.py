import cv2

cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

img = cv2.imread('monkey.jpg')

#Defining minimum window size to be seen as face
skalar = 0.1
minW = img.shape[1]*skalar
minH = img.shape[0]*skalar

minW = int(minW)
minH = int(minH)

print(minW, minH)

#Pre-processing starts
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (minW, minH),
)

#Show rectangles on faces found
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()