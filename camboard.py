#----import----
import numpy as np
import cv2
import imutils
import serial

#----constant----
keyset = [[b'ec',b'ec',b'`',b'`',b'1',b'1',b'2',b'2',b'3',b'3',b'4',b'4',b'5',b'5',b'6',b'6',b'7',b'7',b'8',b'8',b'9',b'9',b'0',b'0',b'-',b'-',b'=',b'=',b'bs',b'bs',b'bs',],
          [b'tb',b'tb',b'tb',b'q',b'q',b'w',b'w',b'e',b'e',b'r',b'r',b't',b't',b'y',b'y',b'u',b'u',b'i',b'i',b'o',b'o',b'p',b'p',b'[',b'[',b']',b']',b'\\',b'\\',b'dl',b'dl',],
          [b'cs',b'cs',b'cs',b'cs',b'a',b'a',b's',b's',b'd',b'd',b'f',b'f',b'g',b'g',b'h',b'h',b'j',b'j',b'k',b'k',b'l',b'l',b';',b';',b'\'',b'\'',b'en',b'en',b'en',b'en',b'en',],
          [b'ls',b'ls',b'ls',b'ls',b'ls',b'z',b'z',b'x',b'x',b'c',b'c',b'v',b'v',b'b',b'b',b'n',b'n',b'm',b'm',b',',b',',b'.',b'.',b'/',b'/',b'up',b'up',b'rs',b'rs',b'rs',b'rs',],
          [b'ct',b'ct',b'ct',b'ct',b'al',b'al',b'al',b'al',b'ha',b'ha',b'ha',b'sp',b'sp',b'sp',b'sp',b'sp',b'sp',b'sp',b'hy',b'hy',b'hy',b'al',b'al',b'lf',b'lf',b'dw',b'dw',b'ri',b'ri',b'ct',b'ct']]
#ec(esc),bs(backspace),tb(tab),dl(delete),cs(caps lock),en(enter),ls(left shift),up(up arror),rs(right shift),fn(function),ct(ctrl),
#wd(window),al(alt),ha(hanja),sp(space bar),hy(hanyoung),ri(right arrow),dw(down arrow),lf(left arrow), dc(document)

WIDTH = 640
HEIGHT = 480

CROP_HEIGHT_START_PERCENTAGE = 50
CROP_HEIGHT_END_PERCENTAGE = 100
CROP_WIDTH_START_PERCENTAGE = 15
CROP_WIDTH_END_PERCENTAGE = 85

#table height, error range
TABLE_HEIGHT = 278
ERROR_RANGE = 5

#피부 경계값
skin_lower = np.array([0,150,100])
skin_upper = np.array([255,190,200])

#input threshold
INPUT_THRESHOLD = 40000

#hsp,hep,wsp,wep = height,width / start,end / percentage
hsp = (int)(HEIGHT*CROP_HEIGHT_START_PERCENTAGE/100)
hep = (int)(HEIGHT*CROP_HEIGHT_END_PERCENTAGE/100)
wsp = (int)(WIDTH*CROP_WIDTH_START_PERCENTAGE/100)
wep = (int)(WIDTH*CROP_WIDTH_END_PERCENTAGE/100)

cam2 = cv2.VideoCapture(2) #2nd floor cam
cam2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cam2.set(3,WIDTH) 
cam2.set(4,HEIGHT)

cam1 = cv2.VideoCapture(0) #1st floor cam
cam1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cam1.set(3,WIDTH) 
cam1.set(4,HEIGHT)



#---varible---
matrix = np.array([[]])

tHight = 1 #tableEdgeHeight
pastInputState = False

#---function---
def affineTransformImage(image_ori,transformMatrix):
    image_crop = image_ori[hsp:hep,wsp:wep]
    image_crop = imutils.rotate_bound(image_crop, 225)
    
    image_affine = cv2.warpPerspective(image_crop,transformMatrix,(WIDTH,HEIGHT))
    
    return image_affine

def getTransformMatrix(image):
    image_crop = image[hsp:hep,wsp:wep]
    #Gary
    image_gray = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
    #edge detection (Canny)
    image_edge = cv2.Canny(image_gray, 30, 50, 7)
    #Rotate
    image_rotated = imutils.rotate_bound(image_edge, 225)
    #find contours, get largest one, get extrime points
    cnts = cv2.findContours(image_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea) 
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[0:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    #affine transform
    aextLeft = np.asarray(extLeft)
    aextRight = np.asarray(extRight)
    aextTop = np.asarray(extTop)
    aextBot = np.asarray(extBot)
    #Option : add extra pixel
    #OFFSET = 0
    #aextLeft += [-OFFSET,0]
    #aextRight += [OFFSET,0]
    #aextTop += [0,-OFFSET]
    #aextBot += [0,OFFSET]
    pts1 = np.float32([aextBot,aextLeft,aextRight,aextTop])
    print(pts1)
    pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    return matrix
    
def getTableHeight(image):
    src = image.copy()    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    canny = cv2.Canny(gray, 150, 300, 7)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 100, maxLineGap = 10)
    print(lines)
    if lines is not None:
        for i in lines:
            tableEdgeHeight = int((lines[0][0][1] + lines[0][0][3])/2)
            print(tableEdgeHeight)
    return tableEdgeHeight
    
def detectInput(image):
    global pastInputState
    image = image[tHight:HEIGHT,wsp:wep] # -ERROR_RANGE
    image_crop = dctSkin(image)
    threshsum = (int)(np.sum(image_crop))
    
    if((not(pastInputState))and(threshsum>INPUT_THRESHOLD)):
        pastInputState = True
        print('on')
        return 1
    elif((pastInputState)and(threshsum<INPUT_THRESHOLD)):
        pastInputState = False
        print('off')
        return 2
    else:
        return 0

def dctSkin(image):
    # BGR -> YCrCb 변환
    YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    # 피부 검출
    mask_hand = cv2.inRange(YCrCb,skin_lower,skin_upper)
    # 피부 색 나오도록 연산
    mask_color = cv2.bitwise_and(image,image,mask=mask_hand)
    return mask_color

def getPoint(image):
    image = dctSkin(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('crop_and_mask2',thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if(len(cnts)>0):
        c = max(cnts, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        return extTop
    else:
        return 0

def calcKey(posW, posH):
    imageDivW = int((31*posW/WIDTH))
    imageDivH = int((6*posH/HEIGHT))
    print('------------------------------')
    print('width = ', posW, 'height = ', posH)
    print('W = ', imageDivW+1, 'th block, H = ', imageDivH+1, 'th block')
    print('your key = ', keyset[imageDivH][imageDivW])
    ser.write(keyset[imageDivH][imageDivW])
    
#----setting----
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    
while(True):
    ret, image2 = cam2.read()
    ret, image1 = cam1.read()
    
    tHight = getTableHeight(image1)
    
    if ((tHight <= TABLE_HEIGHT+ERROR_RANGE)and(tHight >= TABLE_HEIGHT-ERROR_RANGE)):
        matrix = getTransformMatrix(image2)
        cv2.destroyAllWindows()
        print('ready')
        break

#----loop----   
while(True):
    ret, image2 = cam2.read()
    ret, image1 = cam1.read()
    cv2.imshow('image1',image1)
    cv2.imshow('image2',image2)
    
    image2 = affineTransformImage(image2,matrix)
    cv2.imshow('crop',image2)
    
    if(detectInput(image1) == 1): #keyboard on
        #print(getPoint(image2))
        x,y = getPoint(image2)
        if((x<WIDTH+1)and(y<HEIGHT+1)):
            calcKey(x,y)
        else:
            print('confirm size')
        
    elif(detectInput(image1) == 2): #keyboard off
        pass
    else: # nothing happened
        pass
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cam2.release()
        break











