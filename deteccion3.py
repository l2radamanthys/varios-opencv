

import numpy
import cv2
from matplotlib import pyplot
import copy
import sys



def plot(images):
    """
        Plotear las imagenes
    """
    #organizacion de la grilla para mostrar las imagenes
    rows = 4
    cols = 5

    pos = 1
    for img in images:
        pyplot.subplot(rows, cols, pos)
        if len(img) >= 3:
            pyplot.imshow(img[1], img[2])
        else:
            pyplot.imshow(img[1])
        pyplot.title(img[0])
        pyplot.xticks([])
        pyplot.yticks([])
        pos += 1


def save_images(images):
    """
        Guardar las imagenes
    """
    for img in images[1:]: #omito la imagen original
        cv2.imwrite('{}.jpg'.format(img[0]), img[1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])



def main():
    """
    """
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], 0)

    else:
        print "load default image"
        image = cv2.imread('NucleiDAPIconfocal.png', 0)


    #Array de imagenes
    imagenes = [
        #title, image, method
        ['original', image, 'gray'],
    ]

    #imagen a escala de grises
    #si aparece algun mensaje de error en pantalla es por esto
    try:
        print "conversion a gris"
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        #el ejemplo ya estaba en escala de grises :S
        #no conversion
        image_gray = copy.copy(image)
    imagenes.append(['image_gray', image_gray, 'gray'])

    #usar el difuminado gausiano no fue muy practico
    gnucleo = (3, 3)
    gblur_image = cv2.GaussianBlur(image, gnucleo, 0)
    imagenes.append(['gblur_image', gblur_image, 'gray'])

    ret, thresh = cv2.threshold(gblur_image, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagenes.append(['thresh', thresh, 'gray'])

    kernel = numpy.ones((2,2), numpy.uint8) #nucleo
    cl_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
    imagenes.append(['cl_image', cl_image, 'gray'])

    ## quitar puntos internos
    #erosionar imagen
    fg_image = cv2.erode(cl_image, None, iterations=2)
    imagenes.append(['fg_image', fg_image, 'gray'])
    fg_image8 = numpy.uint8(fg_image)

    #dilatacion
    bgt_image = cv2.dilate(cl_image, None, iterations=5)
    ret, bg_image = cv2.threshold(bgt_image, 1, 255, cv2.THRESH_BINARY)

    imagenes.append(['bgt_image', bgt_image, 'gray'])
    imagenes.append(['bg_image', bg_image, 'gray'])

    dist_transform = cv2.distanceTransform(cl_image, cv2.DIST_L2, 5)
    imagenes.append(['dist_transform', dist_transform, 'gray'])

    ret, sure_fg = cv2.threshold(dist_transform, 0.65*dist_transform.max(), 255, 0)
    imagenes.append(['sure_fg', sure_fg, 'gray'])

    sure_fg8 = numpy.uint8(sure_fg)
    ot, contornos, hier = cv2.findContours(sure_fg8, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    imagenes.append(['sure_fg8', sure_fg8, 'gray'])

    dist_transform_b = cv2.distanceTransform(bg_image, cv2.DIST_L2, 5)
    imagenes.append(['dist_transform_b', dist_transform_b, 'gray'])

    ret, sure_fg_b = cv2.threshold(dist_transform_b, 0.65*dist_transform.max(), 255, 0)
    imagenes.append(['sure_fg_b', sure_fg_b, 'gray'])

    sure_fg_b8 = numpy.uint8(sure_fg_b)
    ot, contornos_b, hier = cv2.findContours(sure_fg_b8, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    imagenes.append(['sure_fg_b8', sure_fg_b8, 'gray'])

    #comparo cual fue la imager que mejores resultados proporciono
    ca = len(contornos or [])
    cb = len(contornos_b or [])
    if ca >= cb:
        print "Cantidad de Objectos: {}".format(ca)
    else:
        print "Cantidad de Objectos: {}".format(cb)
        contornos = contornos_b

    circ_img = copy.copy(image)
    num_img = copy.copy(image)
    i = 1
    for cnt in contornos:
        (x, y), rad = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        center2 = (int(x)-10, int(y)-10)
        rad = int(rad)
        if rad < 3: #algunos contornos detectados son muy pequenios
            rad += 6
        cv2.circle(circ_img, center, rad, (0,0,0), -1)
        cv2.putText(num_img, str(i), center2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        i += 1

    imagenes.append(['num_img', num_img, 'gray'])
    imagenes.append(['circ_img', circ_img, 'gray'])


    save_images(imagenes)
    #muestra todas las imagenes
    plot(imagenes)
    pyplot.show()



if __name__ == '__main__':
    main()