
import numpy as np
import cv2

img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('Imagen', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


