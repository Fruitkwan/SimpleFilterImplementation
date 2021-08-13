# Kernel:    Filter Implementation
# Author:    Matthias Dunkel
# Copyright: 2021

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def convulation2d(kernel: np.ndarray, img: Image, func: callable)->np.ndarray:

    assert np.shape(kernel)[0] == np.shape(kernel)[1], "The Kernel must have a size of n x n! Not: " + str(np.shape(kernel))
    assert np.shape(kernel)[0] % 2 == 1, "Ther Kernel size must be odd!"

    kernelSize = np.shape(kernel)[0]

    input    = np.asarray(img)
    padding  = int(np.shape(kernel)[0] / 2)
    
    #RGB or Gray?
    if input.ndim == 3:
        inputPad = np.zeros( (np.shape(input)[0] + 2*padding, np.shape(input)[1] + 2*padding, 3) )
    else:
        inputPad = np.zeros( (np.shape(input)[0] + 2*padding, np.shape(input)[1] + 2*padding) )
    
    inputPad[
            padding:np.shape(input)[0] + padding, 
            padding:np.shape(input)[1] + padding] = input
    
    output = np.zeros_like(input)

    for x in range(np.shape(output)[0]):
        for y in range(np.shape(output)[1]):

            if input.ndim == 3:
                output[x,y] = func(inputPad[x:x+kernelSize, y:y+kernelSize, :])
            else:
                output[x,y] = func(inputPad[x:x+kernelSize, y:y+kernelSize])
    
    return output


def meanBlur(img: Image, size):
    kernel = np.ones((size,size))
    
    def apply(oldimgdata):
        #RGB?
        if oldimgdata.ndim == 3:
            result = np.zeros((1,3))
            for color in range(3):
                result[0,color] = np.sum(oldimgdata[:, :, color]*kernel)
                result[0,color] = int(result[0,color] / np.sum(kernel))
        else:
            result = np.sum(oldimgdata * kernel)        
            result = int(result / np.sum(kernel))
        return result
    
    return Image.fromarray(convulation2d(kernel, img, apply))

def simpleEdgeDetection(img: Image, size: int, treshold: int):
    assert size % 2 != 0, "size must be odd!"

    img = meanBlur(img, 9)

    def apply(oldimgdata):

        def returnValue(result):
            if result >= treshold:
                    return 0
            else:
                return 255

        if oldimgdata.ndim == 3:
            for color in range(3):
                result = 0
                result = np.sum(oldimgdata[:,:,color] * kernel)
                
                return returnValue(result)
        else:
            result = 0
            result = np.sum(oldimgdata * kernel)

            return returnValue(result)
    

    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if x < int(size / 2):
                kernel[x,y] = -1
            elif x > int(size/2):
                kernel[x,y] = 1
    
    horizontaledges = convulation2d(kernel, img, apply)

    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if y < int(size / 2):
                kernel[x,y] = -1
            elif y > int(size/2):
                kernel[x,y] = 1
    
    verticaledges = convulation2d(kernel, img, apply)
    
    for x in range(np.shape(verticaledges)[0]):
        for y in range(np.shape(verticaledges)[1]):
            verticaledges[x,y] = (verticaledges[x,y] + horizontaledges[x,y]) / 2
    
    return Image.fromarray(verticaledges)

# Load Image
imageRGB  = Image.open('../images/test.jpg')
imageGray = imageRGB.convert('L')
#Gray:
fig = plt.figure()
fig.add_subplot(2, 3, 1)
plt.title("Original Image Gray")
plt.imshow(imageGray, cmap='gray')

print('Blurring Image')
fig.add_subplot(2, 3, 2)
imageGrayBlur = meanBlur(imageGray, 17)
plt.title("Blurred Image Gray")
plt.imshow(imageGrayBlur, cmap='gray')

print("Detecting Edges")
fig.add_subplot(2, 3, 3)
imageGrayEdges = simpleEdgeDetection(imageGray, 5, 20)
plt.title("Edge Detection Image Gray")
plt.imshow(imageGrayEdges, cmap='gray')

#RGB
fig.add_subplot(2, 3, 4)
plt.title("Original Image RGB")
plt.imshow(imageRGB)

print("Blurring Image")
imageRGBBlur = meanBlur(imageRGB, 17)
fig.add_subplot(2, 3, 5)
plt.title("Blurred Image RGB")
plt.imshow(imageRGBBlur)

print("Edge Detection Image")
fig.add_subplot(2, 3, 6)
imageRGBEdges = simpleEdgeDetection(imageRGB, 5, 20)
plt.title("Edge Detection Image RGB")
plt.imshow(imageRGBEdges)

plt.show()

