
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

image = cv2.imread('NucleiDAPIconfocal.png', 0)
image2 = copy.copy(image)
#blur_img = cv2.blur(image, (5,5))
blur_img = cv2.GaussianBlur(image, (19, 19), 0)
cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#ret, thresh = cv2.threshold(blur_img, 90, 255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(blur_img, 90, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#invertir imagen
#inv_img = cv2.bitwise_not(thresh)
#rellenar contornos
#contour = cv2.findContours(inv_img, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#for cnt in contour[1]:
#    cv2.drawContours(inv_img, [cnt], 0, 0, 0)
#reinvertir imagen
#thresh = cv2.bitwise_not(inv_img)
thresh = cv2.bitwise_not(thresh)

nucleo = np.ones((2,2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, nucleo, iterations=2)


sure_bg = cv2.dilate(opening, nucleo, iterations=2)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
ret, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 100, 0)

#sure_fg = opening# dist_transform

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
#markers = markers+1

# Now, mark the region of unknown with zero
#markers[unknown==255] = 0

#markers = cv2.watershed(image,markers)
#img[markers == -1] = [255,0,0]

#element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
#thresh2 = cv2.erode(thresha, element)

#markets = cv2.watershed(thresh2,)

circles = cv2.HoughCircles(
        sure_bg,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
        )

#dibujar numeros
font = cv2.FONT_HERSHEY_SIMPLEX
try:
    i = 1
    for c in circles[0,:]:
        cv2.circle(image2, (c[0], c[1]), c[2], (255, 255, 255), -1)
        cv2.putText(image2, str(i), (c[0], c[1]), font, 1, (0,0,0), 2, cv2.LINE_AA)
        i += 1

except:
    pass

plt.subplot(3, 3, 1)
plt.imshow(image, 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 2)
plt.imshow(blur_img, 'gray')
plt.title('Blur')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 3)
plt.imshow(thresh, 'gray')
plt.title('Threshold + fill hole')
plt.xticks([])
plt.yticks([])


plt.subplot(3, 3, 4)
plt.imshow(sure_bg, 'gray')
plt.title('sure bg')
plt.xticks([])
plt.yticks([])


plt.subplot(3, 3, 5)
plt.imshow(sure_fg, 'gray')
plt.title('sure_fg')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 6)
plt.imshow(dist_transform, 'gray')
plt.title('dist tranf')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 7)
plt.imshow(unknown, 'gray')
plt.title('unknown')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 8)
plt.imshow(markers, 'gray')
plt.title('unknown')
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 9)
plt.imshow(image2, 'gray')
plt.title('final')
plt.xticks([])
plt.yticks([])

plt.show()



