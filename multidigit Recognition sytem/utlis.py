import cv2
import numpy as np
from tensorflow.keras.models import load_model


#### READ THE MODEL WEIGHTS
def intializePredectionModel():
    model = load_model('Resources/kagglemodel.h5')
    return model


#### 1 - Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold

def thresh(img):
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold
    

def edge_detect(img):
    blurred_image = cv2.GaussianBlur(img.copy(),(3,3),0)
    edges = cv2.Canny(blurred_image, 70,150)
    dilated = cv2.dilate(edges,(3,3))
    return dilated




#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
# def splitBoxes(img):
#     rows = np.vsplit(img,9)
#     boxes=[]
#     for r in rows:
#         cols= np.hsplit(r,9)
#         for box in cols:
#             boxes.append(box)
#     return boxes

def splitBoxes(img):
    rows = np.vsplit(img,7)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,7)
        for box in cols:
            boxes.append(box)
    # boxes = boxes[7:]
    return boxes


def boxes_by_col(img):
    cols = np.hsplit(img,7)
    boxes=[]
    for col in cols:
        raws= np.vsplit(col,7)
        for box in raws:
            boxes.append(box)
    # boxes = boxes[7:]
    return boxes


def test_slide_image(img):
    t = 6
    img = np.asarray(img)
    # img_5 = img[:img.shape[0] - 3*t, t-4:img.shape[1] -5*t]
    # img = img[t:img.shape[0] - 3*t, t:img.shape[1] -4*t-2]
    # img = img[t:img.shape[0]-3*t, t:img.shape[1]-2*t]
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    cv2.imshow('test_slide',img)

def bounding_digit(img_th):
    t=5
    img_th = remove_boundary(img_th)
    img_th = np.asarray(img_th)
    
    # print(img.shape)
    # img = cv2.resize(img, (28, 28))
    # img = img / 255.0
    # contours, hierarchy = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    # img = cv2.cvtColor(img_th,cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 1) # DRAW ALL DETECTED CONTOURS

    # img = cv2.resize(img, (28, 28))
    
    # opening = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)
    # dilation = cv2.dilate(img_th,kernel,iterations = 1)
    # kernel = np.ones((4,4),np.uint8)
    # img_th = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)
    # img_th =  img_th[t:img_th.shape[0]-t, :img_th.shape[1]-t]
    # kernel = np.ones((2,2),np.uint8)
    # img_th = cv2.erode(img_th,kernel,iterations = 1)        
    # t=5
    # img_th = img_th[t:img_th.shape[0]-t, t:img_th.shape[1]-t]
    # kernel = np.ones((3,3),np.uint8)
    # img_th = cv2.dilate(img_th,kernel,iterations = 1)

    # img_th = cv2.dilate(img_th,kernel,iterations = 1)
    # img_th =  img_th[t:img_th.shape[0]-t, :img_th.shape[1]-t]
    # img_th[:5,:]=0 
    # img_th[-20:,:]=0
    # img_th[:,-10:]=0
    # img_th = img_th[t:img_th.shape[0], t:img_th.shape[1]-t]
    kernel = np.ones((3,3),np.uint8)
    img_th = cv2.medianBlur(img_th,5)
    img_th = cv2.dilate(img_th,kernel,iterations = 1)
    img_th = cv2.resize(img_th, (28, 28))
    cv2.imshow('cont',img_th)


def remove_boundary(image):
    img = image.copy()
    t = 5
    # img[:t,:]=0
    
    img[-t:,:]=0
    img[:,-t:]=0
    # img[:,:t]=0

    # img = img[:img.shape[0]-2, :img.shape[1]-2]

    # for i in range(t,5):
    #     if img[:i,:].any() == True:
    #         img[:i,:]=0
            
    #     else:
    #         break
    # for i in range(t,20):
    #     if img[-i:,:].any() == True:
    #         img[-i:,:]=0
            
    #     else:
    #         break
            
    # # width
    # for i in range(t,10):
    #     if img[:,:i].any() == True:
    #         img[:,:i]=0
            
    #     else:
    #         break

    # # width
    # for i in range(1,10):
    #     if img[:,-i:].any() == True:
    #         img[:,-i:]=0
            
    #     else:
    #         break
    
    return img

def findCenter(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
    result = []
    i=1
    t = 5
    kernel_e = np.ones((1,1),np.uint8)
    kernel_d = np.ones((2,2),np.uint8)

    # kernel_e = np.ones((1,1),np.uint8)

    for image in boxes:
        ## PREPARE IMAGE
        # remove_boundary(image)
        # image=remove_boundary(image)
        img = np.asarray(image)
        
        
        if i=='asda':
            result.append(' ')
            i+=1
            continue
        else:
            
            # cv2.imshow("heloo"+str(i),img)
            # test_img = cv2.bitwise_not(img)
            # test_image = img.copy()
            # cont_img=img.copy()
            # test_image_gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
            # _,thr = cv2.threshold(img,0,100,cv2.THRESH_BINARY_INV) 
            dilated = cv2.dilate(img.copy(),kernel_d,iterations = 1)
            # cv2.imshow("dila"+str(i),dilated)
            # print(dilated.shape)
            # result.append('0')
            # i+=1
            # continue
            color_img = cv2.cvtColor(dilated,cv2.COLOR_GRAY2BGR)
            rec_img = color_img.copy()
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, contours, -1, (0, 255, 0), 2)
            # cv2.imshow("contour img"+str(i),color_img)
            print("number of contours ",len(contours))
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
            digits=[]
            distances = []

            for j, cont in enumerate(sorted_contours[:2],1):
                bl_img=np.zeros(dilated.shape)
                x, y, w, h = cv2.boundingRect(cont)
                cv2.rectangle(rec_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                # calculate distance of rectangle from origin 
                d = ((x-0)**2+(y-0)**2)**(1/2)
                distances.append(d)
                # leng = abs(int(h))
                # pt1 = int(y +( h // 2) - (leng // 2))
                # pt2 = int(x + (w // 2) - (leng // 2))

                # bl_img[y:y+h,x:x+w] = img[y:y+h,x:x+w] 
                # bl_img[pt1:pt1+leng, pt2:pt2+leng] = img[pt1:pt1+leng, pt2:pt2+leng]
                ex_image = img[y:y+h,x:x+w]
                pt1 = findCenter(ex_image.copy())
                pt2 = (bl_img.shape[0]//2,bl_img.shape[1]//2)

                ## (2) Calc offset
                dx = (pt2[0]-pt1[0])
                dy = ( pt2[1] - pt1[1])

                h, w = ex_image.shape[:2]

                
                bl_img[dy:dy + h, dx:dx + w] = ex_image


                # erod = cv2.erode(bl_img,kernel_e,iterations=1)
                dilated = cv2.dilate(bl_img,kernel_d,iterations = 1)

                # cv2.imshow("blank_img_first"+str(i+j),bl_img)
                extracted_image = cv2.resize(dilated , (28, 28))
                
                extracted_image = extracted_image/255.0
                
                
                # extracted_image = cv2.erode(extracted_image,kernel_d)
                # extracted_image=cv2.dilate(extracted_image,kernel_d)
                cv2.imshow("extracted_image"+str(i),extracted_image)

                extracted_image = extracted_image.reshape(1,28,28,1).astype('float32')
                prediction = model.predict(extracted_image)
                classIndex = np.argmax(prediction)
                # print(classIndex)
                probabilityValue = np.amax(prediction)
                ## SAVE TO RESULT
                if probabilityValue > 0.3:
                    digits.append(str(classIndex))
                else:
                    digits.append(' ')
                i+=1

            if distances[0]>distances[1]:
                temp = digits[0]
                digits[0]=digits[1]
                digits[1] = temp


            result.append("".join(digits))
            # cv2.imshow("contour img"+str(i),rec_img)
            # i+=1
            # test_img=img.copy()
            # edged = cv2.Canny(dilated , 30, 200)
            # contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
            # cv2.drawContours(img , contours, 0, (0, 255, 0), 3)
            # cv2.imshow("contour mage"+str(i),dilated)
            # print(len(contours))
            # rectangles = [cv2.boundingRect(ctr) for ctr in contours]
            
            # for rect in rectangles:
            #     bl_img=np.zeros(img.shape)
            #     cv2.rectangle(test_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 3)
            #     leng = abs(int(rect[3]))
            #     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            #     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            #     bl_img[pt1:pt1+leng, pt2:pt2+leng]=img[pt1:pt1+leng, pt2:pt2+leng]
            #     extracted_image =bl_img
            #     # print(extracted_image.shape)
                # cv2.imshow('ex'+str(i),extracted_image)
                # extracted_image = cv2.resize(extracted_image, (28, 28))
                # extracted_image = extracted_image.reshape(1,28,28,1).astype('float32')
                # prediction = np.argmax(model.predict(extracted_image), axis = 1)

                ## GET PREDICTION
                # predictions = model.predict(img)
                # print(predictions)
                # classIndex = np.argmax(prediction)
                # # print(classIndex)
                # probabilityValue = np.amax(prediction)
                # ## SAVE TO RESULT
                # if probabilityValue > 0.3:
                #     digits.append(str(classIndex))
                # else:
                #     digits.append(' ')
            
            
    return result


#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    r_width = 7
    secW = int(img.shape[1]/r_width)
    secH = int(img.shape[0]/r_width)
    for x in range (0,r_width):
        for y in range (0,r_width):
            cv2.putText(img, str(numbers[(x*r_width)+y]),
                        (x*secW+int(secW/2)-20, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2, color, 2, cv2.LINE_AA)
                    # 10 or 1digit
    return img


#### 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    r_width = 7
    secW = int(img.shape[1]/r_width )
    secH = int(img.shape[0]/r_width )
    for i in range (0,r_width ):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver