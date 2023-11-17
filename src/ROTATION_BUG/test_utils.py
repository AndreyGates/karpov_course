'''Modules'''
import utils
from skimage import data

def test_rotated_image():
    '''Test the correctness of an image rotation'''
    image = data.cat() # a cat's image
    width, height, _ = image.shape # the original size

    rotated_image = utils.rotated_image(image)
    new_width, new_height, _ = rotated_image.shape # the rotation size

    assert width == new_width and height == new_height
