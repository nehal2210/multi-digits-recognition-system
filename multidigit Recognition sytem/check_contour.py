
import cv2
import numpy as np
  
# Let's load a simple image with 3 black squares
image = cv2.imread('Resources\\2digit.jpg')
cv2.waitKey(0)
img = cv2.resize(image,(300,300))
big_cont_img = img.copy()
big_cont__rec_img = img.copy()

blank_img = np.zeros(img.shape)
print(blank_img.shape)
# Grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  
# # Find Canny edges
# gray_inverted = cv2.bitwise_not(gray)
# # cv2.waitKey(0)

# _, binary = cv2.threshold(gray_inverted, 125, 255, cv2.THRESH_BINARY)

# cv2.imshow('gray',gray_inverted)
# cv2.imshow('binary',binary)


# # find the contours from the inverted gray-scale image
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # draw all contours
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# print(len(contours))
# cv2.imshow("contour_img",img)





# Blur the image to remove noise
blurred_image = cv2.GaussianBlur(img.copy(),(3,3),0)

cv2.imshow("img",img)
cv2.imshow("blurred_image",blurred_image)

# Apply canny edge detection
edges = cv2.Canny(blurred_image, 70,200)

cv2.imshow("edges",edges)

dilated = cv2.dilate(edges,(3,3))

cv2.imshow("dilate",dilated)

# Detect the contour using the using the edges
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours
# image3_copy = image3.copy()
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow("contour ing",img)

print("Number of contours : ",len(contours))

# big_contour = max(contours, key = cv2.contourArea)
big_contour = contours[0]
sec_big_contour = contours[1]

cv2.drawContours(big_cont_img, big_contour, -1, (0, 255, 0), 2)
cv2.imshow('big cont',big_cont_img)

x, y, w, h = cv2.boundingRect(big_contour)
cv2.rectangle(big_cont__rec_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
cv2.imshow('big_cont__rec_img',big_cont__rec_img)


# paste digit in blank image
blank_img[y:y+h,x:x+w] = img[y:y+h,x:x+w] 
cv2.imshow("blank_img_first",blank_img)
# second biggest

cv2.drawContours(big_cont_img, sec_big_contour, -1, (0, 255, 0), 2)
cv2.imshow(' seond big cont',big_cont_img)

x, y, w, h = cv2.boundingRect(sec_big_contour)
cv2.rectangle(big_cont__rec_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
cv2.imshow('big_cont__rec_img',big_cont__rec_img)
# paste digit in blank image
blank_img[y:y+h,x:x+w] = img[y:y+h,x:x+w] 
cv2.imshow("blank_img_both",blank_img)


# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
# image = cv2.resize(i.copy(),(28,28))
# contours, hierarchy = cv2.findContours(edged, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
  
# print("Number of Contours found = " + str(len(contours)))
  
# Draw all contours
# -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
  
# cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()