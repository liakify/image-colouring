'''
Outputs a 384x384 image named image_cropped_XXXXX.jpg for every image.
'''

from PIL import Image
import numpy as np

'''
Takes in a PIL.Image object and returns a new PIL.Image randomly cropped to the specified dimensions.
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

end = 8189

for i in range(1, end+1):
    im = Image.open('../jpg/image_{:05d}.jpg'.format(i))
    croppedIm = randomCrop(im)
    croppedIm.save('../jpg/image_cropped_{:05d}.jpg'.format(i))
