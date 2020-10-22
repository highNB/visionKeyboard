# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:52:29 2020

@author: PC
"""
keyset = [[ 27, 27, 96, 96, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 48, 48, 45, 45, 61, 61,  8,  8,  8],
          [  9,  9,  9,113,113,119,119,101,101,114,114,116,116,121,121,117,117,105,105,111,111,112,112, 91, 91, 93, 93,124,124,127,127],
          [ 20, 20, 20, 20, 97, 97,115,115,100,100,102,102,103,103,104,104,106,106,107,107,108,108, 59, 59, 39, 39, 13, 13, 13, 13, 13],
          [ 15, 15, 15, 15, 15,122,122,120,120, 99, 99,118,118, 98, 98,110,110,109,109, 44, 44, 46, 46, 47, 47,901,901, 15, 15, 15, 15],
          [905,905,910,910,906,906,911,911,907,907,907, 32, 32, 32, 32, 32, 32, 32,908,908,908,911,911,903,903,902,902,904,904,909,909]]
#enter -13(CR),10(LF) -> 13
#shift -14(SO),15(SI) -> 15
#arrowUP,DOWN,LEFT,RIGHT -> 901(U),902(D),903(L),904(R)
#function -> 905, window -> 906, hanja -> 907, hangul -> 908, document ->909
#ctrl -> 910, alt -> 911


def calcKey(width, height):
    imagePropW = width/IMAGE_WIDTH*100
    imagepropH = height/IMAGE_HEIGHT*100
    imageDivW = (int)(imagePropW/3.226)
    imageDivH = (int)(imagepropH/6)
    print('height = ',height, 'width = ',width)
    print('height = ', imageDivH+1, 'th block, width = ', imageDivW+1, 'th block')
    print('ASCII = ', keyset[imageDivH][imageDivW])
    
    
    
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

while(True):
    height, width = map(int, input('height, width 순서대로 입력').split())
    if((height>IMAGE_HEIGHT) | (width>IMAGE_WIDTH)):
        print('입력사이즈 초과 프로그램 종료')
        break    
    calcKey(width, height)


