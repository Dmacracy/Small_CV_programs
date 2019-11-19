import sys
import os

import numpy as np
import cv2

'''
Created by Daniel McAndrew. 

Purpose:
   Adds an alpha channel to an image that previously did not have one (Converts rbg to rbga).

How To Use:
   Run with two commandline arguments: 
      pathin  = the path to the original 3-channel image
      pathout = the path to the output 4-channel image 
   
Note: 
   The alpha channel for the resulting image will, by default be set to full opacity, 
   but the add_alpha function has an optional argument which changes the applied alpha 
   value. 

   This code can easily be modified to run on every image in a given directory.
   If you need help with this, feel free to contact me.   
 
'''

def get_img(imgName):
  '''Open a file by its input name and return an open cv image object
     which is a uint8 numpy array in bgr space.'''
  image = cv2.imread(imgName)
  return image

def add_alpha(visPath, outpath, alphaval=255, visextension='.jpg'):
  '''Given a path to a set of vis images (of filetype specified by extension), 
     this function will read in all the images and then add an alpha channel 
     to it. The alpha channel will be uniformly valued at alphaval. 
     The output files are written to the outpath directory.'''
  for i in range(len(os.listdir(visPath))):
    img = get_img(os.path.join(visPath, str(i + 1).zfill(4) + visextension))
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alphaval
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    cv2.imwrite(os.path.join(outpath, str(i + 1).zfill(4) + visextension),
                img_BGRA)

if __name__ == '__main__':
  pathin = sys.argv[1]
  pathout = sys.argv[2]
  add_alpha(pathin, pathout)
  
