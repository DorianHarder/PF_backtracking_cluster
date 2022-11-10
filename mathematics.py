import math
import numpy as np


def angle_trunc(a):
  while a < 0.0:
    a += np.pi * 2
  return a

def calc_azimuth(point1, point2):
  '''azimuth between 2 shapely points (interval 0 - 360)'''
  #angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
  x1 = point1[0]
  y1 = point1[1]
  x2 = point2[0]
  y2 = point2[1]

  deltaY = y2 - y1
  deltaX = x2 - x1
  return angle_trunc(math.atan2(deltaY, deltaX))







  #return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 180


#Calculate azimuth function
def azimuthAngle(x1, y1, x2, y2):
  angle = 0.0;
  dx = x2 - x1
  dy = y2 - y1
  if x2 == x1:
    angle = math.pi / 2.0
  if y2 == y1:
    angle = math.pi
  elif y2 < y1:
    angle = 3.0 * math.pi / 2.0
  elif x2 > x1 and y2 > y1:
    angle = math.atan(dx / dy)
  elif x2 > x1 and y2 < y1:
    angle = math.pi / 2 + math.atan(-dy / dx)
  elif x2 < x1 and y2 < y1:
    angle = math.pi + math.atan(dx / dy)
  elif x2 < x1 and y2 > y1:
    angle = math.pi / 2.0 + math.atan(dy / -dx)
  return angle#(angle * 180 / math.pi)
