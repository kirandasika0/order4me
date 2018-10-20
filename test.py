import cv2
cv2.imshow('img', )
ret = cv2.startWindowThread()
print(ret) # if ret is 0, it means that this functionality is not supported by your OpenCV.
cv2.destroyAllWindows()
