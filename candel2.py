#import
import numpy as np
import cv2
import imutils
import copy
import math

#constant

WIDTH = 640
HEIGHT = 480

CROP_HEIGHT_START_PERCENTAGE = 50
CROP_HEIGHT_END_PERCENTAGE = 100
CROP_WIDTH_START_PERCENTAGE = 20
CROP_WIDTH_END_PERCENTAGE = 80

hsp = (int)(HEIGHT*CROP_HEIGHT_START_PERCENTAGE/100)
hep = (int)(HEIGHT*CROP_HEIGHT_END_PERCENTAGE/100)
wsp = (int)(WIDTH*CROP_WIDTH_START_PERCENTAGE/100)
wep = (int)(WIDTH*CROP_WIDTH_END_PERCENTAGE/100)

cap = cv2.VideoCapture(2)

#*********************************************************
#**************************setup**************************
#*********************************************************

def cropImage(image):
    #hsp,hep,wsp,wep = height,width / start,end / percentage
    crop = image[hsp:hep,wsp:wep]
    return crop

while(True):
    #1 - load 
    ret, image_ori = cap.read()
    image_ori = cv2.resize(image_ori, dsize=(640, 480))
    image_crop = cropImage(image_ori)    
    #2 - Gary
    image_gray = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
    #3 - edge detection (Canny)
    image_edge = cv2.Canny(image_gray, 30, 50, 7)
    cv2.imshow("edge dection", image_edge)
    #4 - Rotate
    image_rotated = imutils.rotate_bound(image_edge, 225)
    cv2.imshow("Rotate 45 Degrees", image_rotated)
    
    #5 - find contours, get largest one, get extrime points
    cnts = cv2.findContours(image_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea) 
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[0:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    '''
    image_rotated_color = imutils.rotate_bound(image_crop, 45)
    image_rotated_color2 = image_rotated_color.copy()
    
    위의 두개는 바로 아래에서 이미지에 낙서를 해서 하나더만든거임
    TODO: 아래의 코드는 확인용이므로 지워도됨 그러나 프로젝트 끝날 때 까지는 살림
    draw the outline of the object, then draw each of the
    
    cv2.drawContours(image_rotated_color2, [c], -1, (0, 255, 255), 2)
    cv2.circle(image_rotated_color2, extLeft, 6, (0, 0, 255), -1)
    cv2.circle(image_rotated_color2, extRight, 6, (0, 255, 0), -1)
    cv2.circle(image_rotated_color2, extTop, 6, (255, 0, 0), -1)
    cv2.circle(image_rotated_color2, extBot, 6, (255, 255, 0), -1)
    cv2.imshow("6-contour image", image_rotated_color2)
    '''
    #7 - affine 
    #add extra pixel 
    OFFSET = 0
    aextLeft = np.asarray(extLeft)
    aextRight = np.asarray(extRight)
    aextTop = np.asarray(extTop)
    aextBot = np.asarray(extBot)
    
    aextLeft += [-OFFSET,0]
    aextRight += [OFFSET,0]
    aextTop += [0,-OFFSET]
    aextBot += [0,OFFSET]
    
    pts1 = np.float32([aextBot,aextLeft,aextRight,aextTop])
    pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image_affine = cv2.warpPerspective(image_rotated,matrix,(WIDTH,HEIGHT))    
    
    cv2.imshow("7-affine ", image_affine)

    # Display the resulting frame
    #cv2.imshow("3-blur", image_blur)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


