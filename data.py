'''
Module contains all the data and image processing functions.
'''
import numpy as np
from PIL import Image
from PIL import ImageOps
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from sklearn.neighbors import NearestNeighbors

END = 8189
IMAGE_FOLDER = "../jpg"
SEGMIM_FOLDER = "../segmim"
NEWINPUTS_FOLDER = "../newinputs"
OUTPUT_FOLDER = "../output"

'''
Takes in a PIL.Image object and returns average (R, G, B, Y) values.
Ignores pixels of colour (0, 0, 254) since it is the background colour.
Returns None if the entire image is just the background colour since all pixels are ignored.
'''


def calculateRGBY(image):
    bgColour = (0, 0, 254)
    allPixels = np.array(image.getdata())

    pixelFilter = np.logical_or(
        allPixels[:, 0] != bgColour[0],
        allPixels[:, 1] != bgColour[1],
        allPixels[:, 2] != bgColour[2]
    )
    pixels = allPixels[pixelFilter, :]
    if pixels.shape[0] == 0:
        return None

    rgbSum = np.sum(pixels, axis=0)
    rgbAvg = rgbSum / pixels.shape[0]
    return np.append(rgbAvg, rgbAvg[0] * rgbAvg[1] - rgbAvg[2] ** 2)


'''
Just run this to preprocess the images
'''


def runThis():
    randomCropAllImageAndSegmim()
    createMaskAll()
    applyMaskAll()

'''
Creates mask from segmim image
'''


def createMask(segmim):
    allPixels = np.array(segmim)
    mask = np.logical_and(
        allPixels[:, :, 2] > 250,
        allPixels[:, :, 1] < 2,
        allPixels[:, :, 0] < 2
    )
    return Image.fromarray((mask * 255).astype(np.uint8))


'''
Creates mask from all segmims
'''


def createMaskAll():
    for i in range(1, END + 1):
        im = Image.open('{}/segmim_cropped_{:05d}.jpg'.format(SEGMIM_FOLDER, i))
        mask = createMask(im)
        mask.save('{}/mask_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))


'''
Applies mask to jpg folder
'''


def applyMask(img, mask):
    blackImage = Image.open('../blueImage.jpg')
    invertedMask = ImageOps.invert(mask)
    background = Image.composite(blackImage, img, invertedMask)
    flower = Image.composite(blackImage, img, mask)
    return background, flower


'''
Applies mask to all jpgs
'''


def applyMaskAll():
    for i in range(1, END + 1):
        im = Image.open('{}/image_cropped_{:05d}.jpg'.format(IMAGE_FOLDER, i))
        mask = Image.open('{}/mask_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))
        processedImages = applyMask(im, mask)
        processedImages[0].save('{}/background_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))
        processedImages[1].save('{}/flower_{:05d}.jpg'.format(NEWINPUTS_FOLDER, i))


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
Takes in a PIL.Image object and returns a new PIL.Image randomly cropped to the specified dimensions (w, h).
If unspecified, dim = (384, 384) by default.
If image dimensions are less than dim in any axis then it is not cropped along that axis.
'''


def randomDoubleCrop(image, segmim, dim=(384, 384)):
    w, h = image.size
    left = np.random.randint(0, max(0, w - dim[0]) + 1)
    right = min(w, left + dim[0])
    top = np.random.randint(0, max(0, h - dim[1]) + 1)
    bottom = min(h, top + dim[1])
    return image.crop((left, top, right, bottom)), segmim.crop((left, top, right, bottom))


'''
Outputs a 384x384 image named segmim_cropped_XXXXX.jpg for every segmim.
'''


def randomCropAllImageAndSegmim():
    for i in range(1, END + 1):
        im = Image.open('{}/image_{:05d}.jpg'.format(IMAGE_FOLDER, i))
        segmim = Image.open('{}/segmim_{:05d}.jpg'.format(SEGMIM_FOLDER, i))
        croppedImgs = randomDoubleCrop(im, segmim)
        croppedImgs[0].save('{}/image_cropped_{:05d}.jpg'.format(IMAGE_FOLDER, i))
        croppedImgs[1].save('{}/segmim_cropped_{:05d}.jpg'.format(SEGMIM_FOLDER, i))


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

        x = rgb2lab(img)[:, :, 0]
        y = rgb2lab(img)[:, :, 1:]

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
    sortedIds = np.load("npy/rgbySortedByYellow.npy")[:, 4].astype(int)
    numImages = round(sortedIds.shape[0] * fraction)
    return sortedIds[:numImages]


'''
Generates and saves images produced by the model by combining the original input (L)
    with the predicted output (AB).

L is an (n, width, height, 1) numpy array containing L values of an LAB image.
AB is an (n, width, height, 2) numpy array containing A and B values an LAB image.
ids is a list containing n items for the image IDs to use when saving the images.
'''


def generateImages(L, AB, ids):
    dimensions = L.shape[1:3] + (3,)
    for i in range(len(ids)):
        cur = np.zeros(dimensions)
        cur[:, :, 0] = L[i][:, :, 0]
        cur[:, :, 1:] = AB[i]
        id = ids[i]
        filename = "{}/test_result_{:05d}.jpg".format(OUTPUT_FOLDER, id)
        filenameGray = "{}/test_result_gray_{:05d}.jpg".format(OUTPUT_FOLDER, id)
        rgb = (lab2rgb(cur) * 255).astype(np.uint8)
        gray = (rgb2gray(rgb) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(filename)
        Image.fromarray(gray).save(filenameGray)


'''
Quantizes image AB values into a discrete probability distribution over the most similar colours
    from a specified colour palette (or "bins").

AB is a (width, height, 2) numpy array containing AB values an image.
bins is a (m, 2) numpy array containing m different AB values to quantize to.
k is an int to set the number of nearest bins to quantize to. Default is 5.

Returns a (width, height, m) numpy array containing the per-pixel 
    discrete probability distribution. 
'''


def quantize(AB, bins, k=5):
    numBins = bins.shape[0]
    width, height = AB.shape[:2]
    numPixels = width * height

    nn = NearestNeighbors(n_neighbors=k).fit(bins)
    flatAB = AB.reshape(numPixels, 2)
    dists, indices = nn.kneighbors(flatAB)

    # Using Gaussian distribution for probability values based on distance
    sigma = 5.0
    weights = np.exp(- dists ** 2 / (2 * sigma ** 2))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    result = np.zeros((numPixels, numBins))
    result[np.arange(numPixels)[:, np.newaxis], indices] = weights

    return result.reshape(width, height, numBins)


'''
Convenience function to call quantize() for multiple images at once.

Y is a (n, width, height, 2) numpy array containing AB values of n images.
bins - see quantize().
k - see quantize(). Defaults to 5.

Returns a (n, width, height, m) numpy array where m is the number of colour bins.
'''


def batchQuantize(Y, bins, k=5):
    result = []
    for ab in Y:
        result.append(quantize(ab, bins, k))
    return np.array(result)


'''
Restores image AB values from a discrete probability distribution over the
    specified colour palette (or "bins"). Each pixel value is calculated as the
    expected value of its distribution

prob is a (width, height, m) numpy array containing the probability distribution 
    of each colour value per pixel.
bins is a (m, 2) numpy array containing m different AB values in the specified distribution.
T is a parameter in the interval (0, 1] to adjust the distribution. Default is 0.38.
    T = 1 predicts the mean while T near 0 predicts the mode.
'''


def unquantize(prob, bins, T=0.38):
    adjusted = np.exp(np.log(prob) / T)
    adjusted /= np.sum(adjusted, axis=2)[:, :, np.newaxis]
    return np.dot(adjusted, bins)


'''
Convenience function to call unquantize() for multiple images at once.

Y is a (n, width, height, m) numpy array containing the per pixel probability distribution 
    of n images, where m is the number of colour bins.
bins - see unquantize().
T - see unquantize(). Defaults to 0.38.

Returns a (n, width, height, 2) numpy array of AB values for n images.
'''


def batchUnquantize(Y, bins, T=0.38):
    result = []
    for prob in Y:
        result.append(unquantize(prob, bins, T))
    return np.array(result)
