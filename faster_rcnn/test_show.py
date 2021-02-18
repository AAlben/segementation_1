import os
import cv2


path = '/Users/alben/Downloads/1608010042043.bmp'
img = cv2.imread(path)
box = [88.476105, 105.84817, 548.3718, 441.76297]

start_point = (int(box[0]), int(box[1]))
end_point = (int(box[2]), int(box[3]))
color = (0, 0, 255)
thickness = 2
img = cv2.rectangle(img, start_point, end_point, color, thickness)
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
