

import numpy
import cv2
from matplotlib import pyplot
import copy
import sys



def plot(data):
    pos = 1
    for img in images:
        pyplot.subplot(rows, cols, pos)
        if img.get(2, False):
            pyplot.imshow(img[1], img[2])
        pyplot.title(img[0])
        pyplot.xticks([])
        pyplot.yticks([])
        pos += 1



def main():
    """
    """
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1], 0)

    else:
        image = cv2.imread('NucleiDAPIconfocal.png', 0)

    


    #Array de imagenes     
    imagenes = [
        #title, image
        ['original', image],
        ['RGB', image],
    ]

    plot(imagenes)
    pyplot.show()


if __name__ == '__main__':
    main()