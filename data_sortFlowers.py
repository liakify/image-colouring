'''
Outputs an .npy file storing an np array containing (R, G, B, Y, index) for each image.
'''

from PIL import Image
import numpy as np

'''
Takes in a PIL.Image object and returns average (R, G, B, Y) values.
Ignores pixels of colour (0, 0, 254) since it is the background colour.
Returns None if the entire image is just the background colour since all pixels are ignored.
'''
def calculateRGBY(image):
    bgColour = (0, 0, 254)
    allPixels = np.array(image.getdata())

    pixelFilter = np.logical_or(
        allPixels[:,0] != bgColour[0],
        allPixels[:,1] != bgColour[1],
        allPixels[:,2] != bgColour[2]
    )
    pixels = allPixels[pixelFilter,:]
    if pixels.shape[0] == 0:
        return None

    rgbSum = np.sum(pixels, axis=0)
    rgbAvg = rgbSum / pixels.shape[0]
    return np.append(rgbAvg, rgbAvg[0] * rgbAvg[1] - rgbAvg[2] ** 2)


end = 8189
myList = []

for i in range(1, end+1):
    im = Image.open('../segmim/segmim_{:05d}.jpg'.format(i))
    rgby = calculateRGBY(im)
    if rgby is None:
        continue
    myList.append(np.append(rgby, i))


def sortByYellow(elem):
    return elem[3]

myList.sort(key=sortByYellow, reverse=True)
np.save("npy/rgbySortedByYellow", np.array(myList))
