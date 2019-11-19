import os
import io
import sys
import time
import string
from multiprocessing import Pool

import cv2 as cv
import numpy as np

class Line:
  '''A classs that represents a line in 3D space as an origin 
     point, a direction vector, and a color.'''
  def __init__(self, orig, direct, color):
    self.x = orig[0]
    self.y = orig[1]
    self.z = orig[2]
    self.dx = direct[0]
    self.dy = direct[1]
    self.dz = direct[2]
    self.b = color[0]
    self.g = color[1]
    self.r = color[2]

class BoundingBox:
  '''A class that represents a 3D bounding box.'''
  def __init__(self, xMin, yMin, zMin, xMax, yMax, zMax):
    self.xmin = xMin
    self.ymin = yMin
    self.zmin = zMin
    self.xmax = xMax
    self.ymax = yMax
    self.zmax = zMax

def get_img(imgName):
  '''Open a file by its input name and return an open cv image object
     which is a uint8 numpy array in bgr space.'''
  image = cv.imread(imgName)
  return image

def load_calib_names(path):
  calibNames = [f for f in os.listdir(path) if f.endswith('.txt')]
  return calibNames

def load_img_names(path):
  '''Get a list of the images in a directory that are to be used 
     as network inputs'''
  imgNames = [f for f in os.listdir(path) if f.endswith('.jpg')]
  return imgNames

def load_calibrations(path, calibNames):
  '''Get a list of calibrations by reading in the txt files and 
     returning numpy arrays representing the camera matrices.'''
  calibs = []
  for calibName in calibNames:
    with open(os.path.join(path, "Calibration", calibName)) as calibFile:
      calib = np.loadtxt(calibFile, skiprows=1)
      calibs.append(calib)
  return calibs


def invert_calibs(calibs):
  invCalibs = []
  p4s = []
  for calib in calibs:
    p4s.append(calib[:, 3][np.newaxis].T)
    invCalibs.append(np.linalg.inv(calib[:, 0:3]))
  return invCalibs, p4s
      
def mask_imgs(path, imgNames, maskNames):
  '''Given a list of images and masks, this function applies the masks 
     to the images and saves the resulting masked images.'''    
  for i in range(len(imgNames)):
    img = get_img(os.join(path, "Regular", imgNames[i]))
    mask = get_img(os.path.join(path, "Masks", maskNames[i])) // 255
    masked = img * mask
    cv.imwrite(os.path.join(path, "Masked", imgNames[i]), masked)

def back_proj_lines(path, imgNames, invCalibs, p4s, wParam, densParam):
  lines = []
  for i in range(len(imgNames)):
    maskedImg = get_img(os.path.join(path, "Masked", imgNames[i]))
    orig = np.matmul(invCalibs[i], -1 * p4s[i])
    for y in range(len(maskedImg) // densParam):
      for x in range(len(maskedImg[0]) // densParam):
        if np.sum(maskedImg[y * densParam][x * densParam]) > 0:
          direct = np.matmul(invCalibs[i], [[x * densParam * wParam], [y * densParam * wParam], [wParam]])
          color = maskedImg[y * densParam][x * densParam]
          ln = Line(np.squeeze(orig), np.squeeze(direct), color)
          lines.append(ln)
 #for line in lines:
 #   print(line.x, line.y, line.z)
 #   print(line.dx, line.dy, line.dz)
  return lines

#def bound_lines(lines, minStep, maxStep):
#  tMins = []
#  tMaxes = []
#  for line in lines:
#    use = True
#    txl = (bBox.xmin - line.x) / line.dx
#    txr = (bBox.xmax - line.x) / line.dx
#    tMin = min(txl, txr)
#    tMax = max(txl, txr)
#    tyl = (bBox.ymin - line.y) / line.dy
#    tyr = (bBox.ymax - line.y) / line.dy
#    tyMin = min(tyl, tyr)
#    tyMax = max(tyl, tyr)
#    if (tyMin > tMax or tyMax < tMin):
#      use = False
#    if use:
#      tMin = max(tMin, tyMin)
#      tMax = min(tMax, tyMax)
#      tzl = (bBox.zmin - line.z) / line.dz
#      tzr = (bBox.zmax - line.z) / line.dz
#      tzMin = min(tzl, tzr)
#      tzMax = max(tzl, tzr)
#      if (tzMin > tMax or tzMax < tMin):
#        use = False
#      if use:
#        tMin = max(tMin, tzMin)
#        tMax = min(tMax, tzMax)
#        tMins.append(tMin)
#        tMaxes.append(tMax)
#    if not use:
#      tMins.append(0.0)
#      tMaxes.append(0.0)
#  print(tMins, tMaxes)
#    tMins.append(minStep)
#    tMaxes.append(maxStep)
#  return tMins, tMaxes

def create_points(lines, tMin, tMax, res):
  ptSets = []
  ts = (tMax + tMin) - np.logspace(-np.log10(tMin), -np.log10(tMax), num=res, base=0.1)
  for i in range(len(lines)):    
    xs = lines[i].x + lines[i].dx * 2 * ts
    ys = lines[i].y + lines[i].dy * 2 * ts
    zs = lines[i].z + lines[i].dz * 2 * ts  
    rs = np.repeat(lines[i].r, xs.shape[0])
    gs = np.repeat(lines[i].g, xs.shape[0])
    bs = np.repeat(lines[i].b, xs.shape[0])
    pts = np.stack((xs, ys, zs, rs, gs, bs), axis=1)
    ptSets.append(pts)
  return np.array(ptSets).reshape(len(lines) * int(res), 6)

def write_cloud(path, pts):
  with open(os.path.join(path, "Cloud", "cloud.ply"), "w") as cloudFile:
    cloudFile.write("ply\nformat ascii 1.0\nelement vertex " + str(len(pts)) + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for pt in pts:
      cloudFile.write(str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + " " + str(int(pt[3])) + " " + str(int(pt[4])) + " " + str(int(pt[5])) + "\n")

def mask_cut(path, maskNames, calibs, pts):
  for i in range(len(maskNames)):
    mask = get_img(os.path.join(path, "Masks", maskNames[i]))
    removeIndices = []
    for p in range(len(pts)):
      pt = pts[p]
      projPoint = np.matmul(calibs[i], [[pt[0]], [pt[1]], [pt[2]], [1.0]])
      if projPoint[2] > 0:
        x = int(projPoint[0] / projPoint[2])
        y = int(projPoint[1] / projPoint[2])
      else:
        x = -1.0
        y = -1.0
      if (x < 0 or y < 0 or x >= 5120 or y >= 3072):
        removeIndices.append(p)
      else:
        hitOrMiss = mask[y, x]
        if hitOrMiss[0] == 0:
          removeIndices.append(p)
    pts = np.delete(pts, removeIndices, axis=0)
  return pts
  

if __name__ == '__main__':
  start_time = time.perf_counter()
  path = sys.argv[1]
  imgNames = load_img_names(os.path.join(path, "Regular"))
  maskNames = load_img_names(os.path.join(path, "Masks"))
  #mask_imgs(path, imgNames, maskNames)
  calibNames = load_calib_names(os.path.join(path, "Calibration"))
  calibs = load_calibrations(path, calibNames)
  invCalibs, p4s = invert_calibs(calibs)
  lines = back_proj_lines(path, imgNames,  invCalibs , p4s, 1, 30)
  #lines = back_proj_lines(path, [imgNames[0], imgNames[3], imgNames[8], imgNames[14]],  [invCalibs[0], invCalibs[3], invCalibs[8], invCalibs[14]] , [p4s[0], p4s[3], p4s[8], p4s[14]], 1, 20)
  #tMins, tMaxes = bound_lines(lines, 500.0, 900.0)
  pts = create_points(lines, 550.0, 750.0, 250.0)
  pts = mask_cut(path, [maskNames[0], maskNames[5], maskNames[8], maskNames[10], maskNames[14], maskNames[19]] , [calibs[0], calibs[5], calibs[8], calibs[10], calibs[14], calibs[19]] , pts)
  write_cloud(path, pts)
  print('Time: %g sec' % (time.perf_counter() - start_time))


  
