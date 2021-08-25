# Kernel:    Filter Implementation
# Author:    Matthias Dunkel
# Copyright: 2021

import matplotlib.pyplot as plt
from PIL import Image
from Filter import *
def opt_main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        imageRGB  = Image.open('images/test.jpg')
        imageGray = imageRGB.convert('L')
        MeanBlur(imageGray, 17).execute()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')


def main():
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
    imageGrayBlur = MeanBlur(imageGray, 17).execute()
    plt.title("Blurred Image Gray")
    plt.imshow(imageGrayBlur, cmap='gray')

    print("Detecting Edges")
    fig.add_subplot(2, 3, 3)
    imageGrayEdges = SimpleEdgeDetection(imageGray, 5, 20).execute()
    plt.title("Edge Detection Image Gray")
    plt.imshow(imageGrayEdges, cmap='gray')

    #RGB
    fig.add_subplot(2, 3, 4)
    plt.title("Original Image RGB")
    plt.imshow(imageRGB)

    print("Blurring Image")
    imageRGBBlur = MeanBlur(imageRGB, 17).execute()
    fig.add_subplot(2, 3, 5)
    plt.title("Blurred Image RGB")
    plt.imshow(imageRGBBlur)

    print("Edge Detection Image")
    fig.add_subplot(2, 3, 6)
    imageRGBEdges = SimpleEdgeDetection(imageRGB, 5, 20).execute()
    plt.title("Edge Detection Image RGB")
    plt.imshow(imageRGBEdges)
    plt.show()

if __name__ == "__main__":
    opt_main()
