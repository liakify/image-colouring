'''
Module contains all the data preprocessing functions.
'''
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from tensorflow.keras.preprocessing.image import load_img, img_to_array

END = 8189
IMAGE_FOLDER = "../jpg"

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

'''
Outputs an .npy file storing an np array containing (R, G, B, Y, index) for each image, sorted by Y (decreasing).
'''
def sortImagesByYellow():
    myList = []

    for i in range(1, END + 1):
        im = Image.open('../segmim/segmim_{:05d}.jpg'.format(i))
        rgby = calculateRGBY(im)
        if rgby is None:
            continue
        myList.append(np.append(rgby, i))


    def sortByYellow(elem):
        return elem[3]

    myList.sort(key=sortByYellow, reverse=True)
    myList = np.array(myList)
    np.save("npy/rgbySortedByYellow", myList)
    return myList

'''
Takes in a PIL.Image object and returns a new PIL.Image randomly cropped to the specified dimensions (w, h).
If unspecified, dim = (384, 384) by default.
If image dimensions are less than dim in any axis then it is not cropped along that axis.
'''
def randomCrop(image, dim=(384, 384)):
    w, h = image.size
    left = np.random.randint(0, max(0, w - dim[0]) + 1)
    right = min(w, left + dim[0])
    top = np.random.randint(0, max(0, h - dim[1]) + 1)
    bottom = min(h, top + dim[1])
    return image.crop((left, top, right, bottom))

'''
Outputs a 384x384 image named image_cropped_XXXXX.jpg for every image.
'''
def randomCropAll():
    for i in range(1, END + 1):
        im = Image.open('{}/image_{:05d}.jpg'.format(IMAGE_FOLDER, i))
        croppedIm = randomCrop(im)
        croppedIm.save('{}/image_cropped_{:05d}.jpg'.format(IMAGE_FOLDER, i))

'''
Loads images based on the list of image IDs specified.

Returns 2 numpy arrays X, Y.
X is a (n, width, height, 1) array containing the L values in the LAB space.
Y is a (n, width, height, 2) array containing the A, B values in the LAB space.
    n refers to the number of images loaded
'''
def loadImageData(ids):
    X = []
    Y = []
    for i in ids:
        img = Image.open('{}/image_cropped_{:05d}.jpg'.format(IMAGE_FOLDER, i))
        img = np.array(img)
        
        x = rgb2lab(img)[:,:,0]
        y = rgb2lab(img)[:,:,1:]
        y /= 128

        X.append(x.reshape(x.shape + (1,)))
        Y.append(y)

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y


'''
Prepares a list of image IDs to use in training and testing based on the sorted order of IDs stored in the .npy file.
The .npy file is presumably already sorted by yellow when it was saved.

"fraction" should be a number between 0 and 1 representing the fraction of images to take.
For our project, we use 0.1 for easy, 0.5 for medium and 1.0 for hard,
    e.g. easy => take the 10% most yellow images
'''
def getImageIds(fraction):
    sortedIds = np.load("npy/rgbySortedByYellow.npy")[:,4].astype(int)
    numImages = round(sortedIds.shape[0] * fraction)
    return sortedIds[:numImages]
