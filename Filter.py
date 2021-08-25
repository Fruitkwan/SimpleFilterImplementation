import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Filter():
    """The Filter Class is an abstract that encapsulate basic mehtods for a filter. All Filters need to implement the apply function
    
    ### Methods:

    __init: -> Filter

    apply: -> np.ndarray

    convulation2d: -> np.ndarray

    execute: -> Image
    """

    def __init__(self, img: Image, size: int):
        assert size > 0, "The Kernel size must be positive."
        assert size == size, "The Kernel must have a size of n x n! Not: " + str(size)
        
        if size % 2 == 0:
            size += 1
        self.size   = size
        self.img    = img
        self.kernel =  np.zeros((size, size))

    def convolution2d(self, kernel: np.ndarray, img: Image, func: callable):
        """ Applys a function on an given image with a given kernel
    
        ### Arguments:

        kernel: np.ndarray An quadratic numpy Array with an odd size. If not it throws an AssertionError

        img: Image An Pillow Image. The Image should be either a GrayScale Image or an RGB Image

        func: callable the function of the kernel.

        ### Returns

        np.ndarray : The Result Image as an Array for further work. Note it does not return a new image, 
                    so more filter can be applied withour transforming it back to an numpy array 
        """
        input      = np.asarray(img)
        inputShape = np.shape(input)
        padding    = int(self.size / 2)
    
        if input.ndim == 3: # RGB 
            inputPad = np.zeros( (inputShape[0] + 2*padding, inputShape[1] + 2*padding, 3) )
        else: # Grey Scale
            inputPad = np.zeros( (inputShape[0] + 2*padding, inputShape[1] + 2*padding) )
    
        inputPad[
                padding:inputShape[0] + padding, 
                padding:inputShape[1] + padding] = input
    
        output = np.zeros_like(input)

        for x in range(inputShape[0]):
            for y in range(inputShape[1]):

                if input.ndim == 3:
                    output[x,y] = func(inputPad[x:x+self.size, y:y+self.size, :])
                else:
                    output[x,y] = func(inputPad[x:x+self.size, y:y+self.size])
        
        return output

    def apply(self)->np.ndarray:
        raise NotImplemented

    def execute(self)->Image:
        return Image.fromarray( self.convolution2d(self.kernel, self.img, self.apply) )

class MeanBlur(Filter):
    """meanBlurs the given Image.
    
    ### Arguments:

    img: Image the image that should be blurred

    size: int the size of the kernel. If not odd. It will incremented by one.
    
    ### Returns

    Image: The Blurred Image
    """
    def __init__(self, img: Image, size: int):
        super().__init__(img, size)
        self.kernel = np.ones((self.size,self.size))
        self.kernelSum = np.sum(self.kernel)
        self.rgbresult = np.zeros((1,3))

    def apply(self, oldimgdata: np.ndarray)->np.ndarray:
        #RGB?
        if oldimgdata.ndim == 3:
            result = self.rgbresult
            for color in range(3):
                result[0,color] = np.sum( np.matmul (oldimgdata[:, :, color], self.kernel ))
                result[0,color] = int(result[0,color] / self.kernelSum)
        else:
            result = np.sum( np.matmul(oldimgdata, self.kernel) )        
            result = int(result / self.kernelSum)
        return result

class SimpleEdgeDetection(Filter):

    def __init__(self, img: Image, size: int, treshold: int):
        super().__init__( MeanBlur(self.img, self.size).execute() , size)
        self.treshold = treshold
    
    def apply(self, oldimgdata: np.ndarray)->np.ndarray:
        if oldimgdata.ndim == 3:
            for color in range(3):
                result = 0
                result = np.sum(np.matmul(oldimgdata[:,:,color], self.kernel))
        else:
            result = 0
            result = np.sum(np.matmul(oldimgdata,  self.kernel))

        if result >= self.treshold:
            return 0
        else:
            return 255

    def execute(self)->Image:
        for x in range(self.size):
            for y in range(self.size):
                if x < int(self.size / 2):
                    self.kernel[x,y] = -1
                elif x > int(self.size/2):
                    self.kernel[x,y] = 1
        horizontaledges = self.convolution2d(self.kernel, self.img, self.apply)

        for x in range(self.size):
            for y in range(self.size):
                if y < int(self.size / 2):
                    self.kernel[x,y] = -1
                elif y > int(self.size/2):
                    self.kernel[x,y] = 1
        verticaledges = self.convolution2d(self.kernel, self.img, self.apply)

        for x in range(np.shape(verticaledges)[0]):
            for y in range(np.shape(verticaledges)[1]):
                verticaledges[x,y] = (verticaledges[x,y] + horizontaledges[x,y]) / 2
        
        return Image.fromarray(verticaledges)
