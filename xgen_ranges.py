import numpy as np
import string
import sys
import os

def xgen_ranges(xMin, yMin, zMin, xMax, yMax, zMax, numXDivs, numYDivs, numZDivs):
  xs = np.linspace(xMin, xMax, numXDivs + 1)
  ys = np.linspace(yMin, yMax, numYDivs + 1)
  zs = np.linspace(zMin, zMax, numZDivs + 1)
  corners = []
  thing = 1
  for i in range(numXDivs):
    xLo = xs[i]
    xHi = xs[i + 1]
    for j in range(numYDivs):
      yLo = ys[j]
      yHi = ys[j + 1]
      for k in range(numZDivs):
        zLo = zs[k]
        zHi = zs[k + 1]
        corners.append(np.array([xLo, yLo, zLo, xHi, yHi, zHi]))
        thing += 1
  return corners

def load_calib_names(path):
  calibNames = [f for f in os.listdir(path) if f.endswith('.txt')]
  return calibNames

def load_calibrations(path, calibNames):
  '''Get a list of calibrations by reading in the txt files and 
     returning numpy arrays representing the camera matrices.'''
  calibs = []
  for calibName in calibNames:
    with open(os.path.join(path, calibName)) as calibFile:
      calib = np.loadtxt(calibFile, skiprows=1)
      calibs.append(calib)
  return calibs

def region_midpoint(corner):
  '''Find the midpoint of a 3D bounding box.'''
  xMin = corner[0]
  yMin = corner[1]
  zMin = corner[2]
  xMax = corner[3]
  yMax = corner[4]
  zMax = corner[5]
  return np.array([xMax - xMin / 2.0,  yMax - yMin / 2.0, zMax - zMin / 2.0])


def visible_points(calibs, corners):
  '''For each region, create a list of cameras that can see that region.'''
  for p in range(len(corners)):
    # check if three of the four of the corners of the XZ plane of the bounding box of the region are visible
    # by each camera and use that as a criterion for creating a list of which cameras can see a given region. 
    pt1 = np.array([corners[p][0], corners[p][1], corners[p][2]])
    pt2 = np.array([corners[p][0], corners[p][1], corners[p][5]])
    pt3 = np.array([corners[p][3], corners[p][1], corners[p][2]])
    pt4 = np.array([corners[p][3], corners[p][1], corners[p][5]])
    camsCanSee = []
    for i in range(len(calibs) - 2):
      seenpts = 0
      projPoint1 = np.matmul(calibs[i], [[pt1[0]], [pt1[1]], [pt1[2]], [1.0]])
      projPoint2 = np.matmul(calibs[i], [[pt2[0]], [pt2[1]], [pt2[2]], [1.0]])
      projPoint3 = np.matmul(calibs[i], [[pt3[0]], [pt3[1]], [pt3[2]], [1.0]])
      projPoint4 = np.matmul(calibs[i], [[pt4[0]], [pt4[1]], [pt4[2]], [1.0]])
      projPoints = [projPoint1, projPoint2, projPoint3, projPoint4]
      for projPoint in projPoints:
        if projPoint[2] > 0:
          x = int(projPoint[0] / projPoint[2])
          y = int(projPoint[1] / projPoint[2])
        else:
          x = -1.0
          y = -1.0
        if (x > 0 and y > 0 and x < 5120 and y < 3072):
          seenpts += 1
      if seenpts >= 4:
        camsCanSee.append(i + 1)
    print('<Parameter ParameterName="XgenMainBoxCorners' + str(2 * p + 1) + '" ParameterValue="' + str(corners[p][0] - 5) + ',' + str(corners[p][1]) + ',' + str(corners[p][2] - 5) + ',' + str(corners[p][3] + 5) + ',' + str(corners[p][4]) + ',' + str(corners[p][5] + 5) + '"/>')
    print('<Parameter ParameterName="XgenMainTargetImages' + str(2 * p + 1)  + '" ParameterValue="' + ','.join(str(i) for i in camsCanSee) + '"/>')
    print('<Parameter ParameterName="XgenMainBoxCorners' + str(2 * p + 2) + '" ParameterValue="' + str(corners[p][0] - 5) + ',' + str(corners[p][1]) + ',' + str(corners[p][2] - 5) + ',' + str(corners[p][3] + 5) + ',' + str(corners[p][4]) + ',' + str(corners[p][5] + 5) + '"/>')
    print('<Parameter ParameterName="XgenMainTargetImages' + str(2 * p + 2)  + '" ParameterValue="' + ','.join(str(i) for i in camsCanSee) + '"/>')

def lowres_visible_points(calibs, corners):
  '''For each region, create a list of cameras that can see that region.'''
  a = 0
  b = 0
  for p in range(len(corners)):
    # check if three of the four of the corners of the XZ plane of the bounding box of the region are visible
    # by each camera and use that as a criterion for creating a list of which cameras can see a given region. 
    pt1 = np.array([corners[p][0], corners[p][1], corners[p][2]])
    pt2 = np.array([corners[p][0], corners[p][1], corners[p][5]])
    pt3 = np.array([corners[p][3], corners[p][1], corners[p][2]])
    pt4 = np.array([corners[p][3], corners[p][1], corners[p][5]])
    camsCanSee = []
    for i in range(len(calibs) - 2):
      seenpts = 0
      projPoint1 = np.matmul(calibs[i], [[pt1[0]], [pt1[1]], [pt1[2]], [1.0]])
      projPoint2 = np.matmul(calibs[i], [[pt2[0]], [pt2[1]], [pt2[2]], [1.0]])
      projPoint3 = np.matmul(calibs[i], [[pt3[0]], [pt3[1]], [pt3[2]], [1.0]])
      projPoint4 = np.matmul(calibs[i], [[pt4[0]], [pt4[1]], [pt4[2]], [1.0]])
      projPoints = [projPoint1, projPoint2, projPoint3, projPoint4]
      for projPoint in projPoints:
        if projPoint[2] > 0:
          x = int(projPoint[0] / projPoint[2])
          y = int(projPoint[1] / projPoint[2])
        else:
          x = -1.0
          y = -1.0
        if (x > 0 and y > 0 and x < 5120 and y < 3072):
          seenpts += 1
      if seenpts >= 3:
        camsCanSee.append(i + 1)
    if p % 6 < 3:
      a += 1
      print('<Parameter ParameterName="BoxCornersLowResA' + str(a) + '" ParameterValue="' + str(corners[p][0]) + ',' + str(corners[p][1]) + ',' + str(corners[p][2]) + ',' + str(corners[p][3]) + ',' + str(corners[p][4]) + ',' + str(corners[p][5]) + '"/>')
      print('<Parameter ParameterName="TargetImagesLowResA' + str(a)  + '" ParameterValue="' + ','.join(str(i) for i in camsCanSee) + '"/>')
    else:
      b += 1
      print('<Parameter ParameterName="BoxCornersLowResB' + str(b) + '" ParameterValue="' + str(corners[p][0]) + ',' + str(corners[p][1]) + ',' + str(corners[p][2]) + ',' + str(corners[p][3]) + ',' + str(corners[p][4]) + ',' + str(corners[p][5]) + '"/>')
      print('<Parameter ParameterName="TargetImagesLowResB' + str(b) + '" ParameterValue="' + ','.join(str(i) for i in camsCanSee) + '"/>')
      


if __name__ == '__main__':
  #print("<Parameters>")
  corners = xgen_ranges(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]))
  path = sys.argv[10]
  calibs = load_calibrations(path, load_calib_names(path))
  visible_points(calibs, corners)
  #print("</Parameters>")
