import numpy as np
import cv2

img = cv2.imread("5.jpg")
cv2.imshow('RGB', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert to HSV and show image in HSV
hsv_img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv_img)
#divide into HUE, SATURATION AND VALUE
h = hsv_img[:, :, 0]
cv2.imshow("HUE", h)
s = hsv_img[:, :, 1] #tuoi
cv2.imshow("SATURATION", s)
v = hsv_img[:, :, 2] #sang
cv2.imshow("VALUE", v)
#close opened windows
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.calcHist([v],[0],None,[256],[0,256])    

v_equalized = cv2.equalizeHist(v)
h_equalized = cv2.equalizeHist(h)
s_equalized = cv2.equalizeHist(s)
cv2.imwrite("v_equalized.png", v_equalized)
cv2.imwrite("s_equalized.png",s_equalized)

#merge h,s,v into HSV image
# HSV_img = cv2.merge([h,s,v])
HSV_img = cv2.merge([h,s_equalized,v_equalized])
# cv2.imwrite("merged.png", HSV_img)
cv2.imshow("Merged HSV", HSV_img)

#Convert HSV into BGR
BGR_img = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
# cv2.imwrite("BGR.png", BGR_img)

cv2.imshow("BGR", BGR_img)
cv2.imshow("original", img)


#close opened windows
cv2.waitKey(0)
cv2.destroyAllWindows()


