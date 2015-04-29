

import numpy
import cv2
from matplotlib import pyplot
import copy


#cargar imagenes
image = cv2.imread('NucleiDAPIconfocal.png')
#convertir a escala de grises
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#obtiene mascaras
ret, thresh = cv2.threshold(image_gray, 90, 255, cv2.THRESH_BINARY)
#gaus_thresh = cv2.adaptiveThreshold(image_gray, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#filtro gaussiano para quitar ruido
gnucleo = (5, 5)
gblur = cv2.GaussianBlur(image, gnucleo, 0)

#nueva mascara con reduccion de ruido
ret, ob_thresh = cv2.threshold(gblur, 80, 255, cv2.THRESH_BINARY)

cl_nucleo = numpy.ones((3, 3), numpy.uint8)
closing = cv2.morphologyEx(ob_thresh, cv2.MORPH_CLOSE, cl_nucleo)

#gr_nucleo = numpy.ones((5, 5), numpy.uint8)
#gradient = cv2.morphologyEx(ob_thresh, cv2.MORPH_GRADIENT, gr_nucleo)

#erosionar
er_nucleo = numpy.ones((3, 3), numpy.uint8)
#erosion = cv2.erode(ob_thresh, er_nucleo, iterations=4)
ero_clos = cv2.erode(closing, er_nucleo, iterations=4)

ero_clos_g = cv2.cvtColor(ero_clos, cv2.COLOR_BGR2GRAY)

transform = cv2.distanceTransform(ero_clos_g, cv2.DIST_L2, 0)
ret, tr_thresh = cv2.threshold(transform, 0.3*transform.max(), 255, cv2.THRESH_BINARY)
#tr_thresh_g = cv2.cvtColor(tr_thresh, cv2.COLOR_BGR2GRAY)

ot, contornos, hier = cv2.findContours(ero_clos_g, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#ot, contornos, hier = cv2.findContours(tr_thresh_g, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
print len(contornos) or 0

circ_img = copy.copy(image)
for cnt in contornos:
    (x, y), rad = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    rad = int(rad)
    cv2.circle(circ_img, center, rad, (255,255,255), 3)
    

#imagenes a mostrar
images = [
    #title, image
    ['image', image],
    ['image_gray', image_gray],
    ['thresh', thresh],
    ['gblur', gblur],
    ['ob_thresh', ob_thresh],
    #['closing', closing],
    #['gradient', gradient],
    #['erosion', erosion],
    ['ero_clos', ero_clos],
    ['ero_clos_g', ero_clos_g],
    ['transform', transform],
    ['transform', tr_thresh],
    ['circ_img', circ_img],

]

#organizacion vista 
rows = 2
cols = 5

#mostrar las imagenes
pos = 1
for img in images:
    pyplot.subplot(rows, cols, pos)
    pyplot.imshow(img[1], 'gray')
    pyplot.title(img[0])
    pyplot.xticks([])
    pyplot.yticks([])
    pos += 1

pyplot.show()


