import os
import sys
import pickle
import string

import cv2 as cv
import numpy as np

def get_img(img_name):
  '''Open a file by its input name and return an open cv image object
     which is a uint8 numpy array in bgr space.'''
  image = cv.imread(img_name)
  return image

def load_img_names(path):
  '''Get a list of the images in a directory that are to be used 
     as network inputs'''
  img_names = [f for f in os.listdir(path) if f.endswith('.jpg')]
  return img_names

def subtract_sets(path1, im1_names, path2, im2_names):
  im_diffs = []
  for i in range(len(im1_names)):
    im1 = cv.cvtColor(get_img(path1 + im1_names[i]), cv.COLOR_BGR2HLS)
    im2 = cv.cvtColor(get_img(path2 + im2_names[i]), cv.COLOR_BGR2HLS) 
    im_diff = cv.absdiff(im1 , im2)
    im_diffs.append(im_diff)
  return im_diffs

def threshold(imgs, thresh, kernel):
  thresholdeds = []
  for img in imgs:
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rtv, thresholded = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)
    thresholded = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)
    thresholdeds.append(thresholded)
  return thresholdeds

def remove_color(path, img_names, color, color_width):
  '''Given a path to a directory of images, 
     the names of the images, a bgr color tuple, 
     and a bgr tolerance tuple, this code masks out 
     every pixel in the images that fall within the color 
     range defined by the specified color +/- the color width.'''
  b_low = max(color[0] - color_width[0], 0)
  b_hi  = min(color[0] + color_width[0], 255)
  g_low = max(color[1] - color_width[1], 0)
  g_hi  = min(color[1] + color_width[1], 255)
  r_low = max(color[2] - color_width[2], 0)
  r_hi  = min(color[2] + color_width[2], 255)
  lower = np.array([b_low, g_low, r_low])
  upper = np.array([b_hi, g_hi, r_hi])
  
  for i in range(len(img_names)):
    img = get_img(path + img_names[i])
    mask = cv.inRange(img, lower, upper)
    invert_mask = cv.bitwise_not(mask)
    out_img = cv.bitwise_and(img, img, mask = invert_mask)
    cv.imwrite(path + 'removed\\' + img_names[i], out_img)


def write_imgs(path, img_names, imgs):
  for i in range(len(imgs)):
    cv.imwrite(path + img_names[i], imgs[i])  

if __name__ == '__main__':
  path_in1 = sys.argv[1]
  path_in2 = sys.argv[2]
  path_out = sys.argv[3]
  thresh = float(sys.argv[4])
  kernel = np.ones((int(sys.argv[5]),int(sys.argv[5])), np.uint8)
  img1_names = load_img_names(path_in1)
  img2_names = load_img_names(path_in2)

  color = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])]
  color_width = [int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11])]
  remove_color(path_in1, img1_names, color, color_width)
  remove_color(path_in2, img2_names, color, color_width)

  # img1_rem_color_names = load_img_names(path_in1 + 'removed\\')
  # img2_rem_color_names = load_img_names(path_in2 + 'removed\\')
  #
  # img_diffs = subtract_sets(path_in1, img1_names, path_in2, img2_names)
  # threshed = threshold(img_diffs, thresh, kernel)
  # write_imgs(path_out, img1_names, threshed)
