print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
import sudukoSolver

########################################################################
pathImage = "Resources/mldigit.jpg"
heightImg = 420 #434
widthImg = 420
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################


#### 1. PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

# #### 2. FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
# print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

#  my steps
    # digi_image = thresh(imgWarpColored)
    digi_image = edge_detect(imgWarpColored)

    digi_image = digi_image[55:-60,10:]  # 120 by 120 for previous image
    digi_image = cv2.resize(digi_image, (widthImg, heightImg))
 
    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    # imgSolvedDigits = imgBlank.copy()
    # boxes = splitBoxes(digi_image)  #imgWarpColored
    # print(len(boxes))
    # cv2.imshow("Sample_boxes",boxes[7])
    # test_slide_image(boxes[24])
    box_col = boxes_by_col(digi_image) 
    # test_slide_image(box_col[1])

    # bound_img = imgWarpColored[110:,110:]
    # bound_img = cv2.resize(bound_img, (widthImg, heightImg))
    # bounding_digit(box_col[41])
    # cv2.imshow("Sample",box_col[9])
    numbers = getPredection(box_col, model)
    # numbers = getPredection(boxes, model)
   
    print(numbers)
    # print(len(box_col))
    # you can show this
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    # you can show this
    grid_image = drawGrid(imgDetectedDigits)
    # numbers = np.asarray(numbers)
    # posArray = np.where(numbers > 0, 0, 1)
    # print(posArray)


    # #### 5. FIND SOLUTION OF THE BOARD
    # board = np.array_split(numbers,9)
    # print(board)
    # try:
    #     sudukoSolver.solve(board)
    # except:
    #     pass
    # print(board)
    # flatList = []
    # for sublist in board:
    #     for item in sublist:
    #         flatList.append(item)
    # solvedNumbers =flatList*posArray
    # imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # #### 6. OVERLAY SOLUTION
    # pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    # pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    # imgInvWarpColored = img.copy()
    # imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    # inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    # imgDetectedDigits = drawGrid(imgDetectedDigits)
    # imgSolvedDigits = drawGrid(imgSolvedDigits)

    # imageArray = ([img,imgThreshold,imgContours, imgBigContour],
    #               [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])

    imageArray = ([imgWarpColored, digi_image,grid_image],[img,imgThreshold,imgContours])
   
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv2.waitKey(0)

