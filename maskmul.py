import sys
import os

import numpy as np
import cv2 as cv

'''
Created by Daniel McAndrew. 

Purpose:
   Applies a set of masks to a corresponding set of vis images and 
   saves the resulting masked images.

How To Use:
   Run with two commandline arguments: 
      visPath  = the path to the vis image directory
      maskPath = the path to the mask directory
      outPath  = the path to the output directory 
   
Note: 
   The input masks are assumed to be single-channel 8-bit images.  
'''

def get_img(imgName):
  '''Open a file by its input name and return an open cv image object
     which is a uint8 numpy array in bgr space.'''
  image = cv.imread(imgName)
  return image

def mask_imgs(visPath, maskPath, outPath):
  '''Given a directory containing vis images a directory containing masks, 
     and an output directory, this function applies the masks to the images 
     and saves the resulting masked images in the output directory. The 
     function assumes that the image names of the masks and vis images will be
     '0001.jpg', '0002.jpg', etc. and will fail otherwise. '''    
  for i in range(len(os.listdir(visPath))):
    img = get_img(os.path.join(visPath, str(i + 1).zfill(4) + '.jpg'))
    mask = get_img(os.path.join(maskPath, str(i + 1).zfill(4) + '.jpg')) / 255
    masked = img * mask
    cv.imwrite(os.path.join(outPath, str(i + 1).zfill(4) + '.jpg'), masked)

if __name__ == '__main__':
  visPath = sys.argv[1]
  maskPath = sys.argv[2]
  outPath = sys.argv[3]
  mask_imgs(visPath, maskPath, outPath)
