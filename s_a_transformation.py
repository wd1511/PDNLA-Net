from PIL import Image,  ImageEnhance
from pylab import array, shape, tanh
from scipy.ndimage import filters
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import thinplate as tps
import cv2
import random
import math
import torch

# Python tps https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/pytorch.py
# Reference : https://hobbydev.tistory.com/56
# https://github.com/WarBean/tps_stn_pytorch tps stn pytorch

def get_outline_image(img_path):
    gamma = 0.97
    phi = 200
    epsilon = 0.1
    k = 2.5
    sigma = 0.7
    im = Image.open(img_path).convert('L')
    im = array(ImageEnhance.Sharpness(im).enhance(3.0))
    im2 = filters.gaussian_filter(im, sigma)
    im3 = filters.gaussian_filter(im, sigma * k)
    differencedIm2 = im2 - (gamma * im3)
    (x, y) = shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + tanh(phi * (differencedIm2[i, j]))

    gray_pic = differencedIm2.astype(np.uint8)
    final_img = Image.fromarray(gray_pic)
    return final_img

# https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    print('image : ', id(image))
    if random_state is None:
        random_state = np.random.RandomState(None)

        # print(random_state)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='nearest')  # wrap,reflect, nearest
    return distored_image.reshape(image.shape)

# Reference : https://github.com/cheind/py-thin-plate-spline

def tps_transform(img, dshape=None):

    while True:
        point1 = round(random.uniform(0.3, 0.7), 2)
        point2 = round(random.uniform(0.3, 0.7), 2)
        range_1 = round(random.uniform(-0.25, 0.25), 2)
        range_2 = round(random.uniform(-0.25, 0.25), 2)
        if math.isclose(point1 + range_1, point2 + range_2):
            continue
        else:
            break

    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1, point1],
        [point2, point2],
    ])

    c_dst = np.array([
        [0., 0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1 + range_1, point1 + range_1],
        [point2 + range_2, point2 + range_2],
    ])

    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    img_out = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
    return img_out


def tps_torch(img_in,dshape=None):
    img = img_in.numpy()
    B,C,H,W = img.shape
    for i in range(B):
        img_now = img[i].transpose(1, 2, 0)
        img_now = tps_transform(img_now,dshape)
        img[i] = img_now.transpose(2, 0, 1)
    img[img > 1] = 1
    img[img < -1] = -1
    return torch.from_numpy(img)



def add_color_noise(ref):
    reference = ref.numpy()
    reference = (reference + 1)/2*255
    noise = np.random.uniform(-1, 1, np.shape(reference)).astype(np.float32)
    #noise = noise / 255.0 * 2.0
    noise = noise * 50.0
    reference = np.array(reference) + noise
    reference = np.clip(reference,0,255)
    reference = reference/255 * 2 - 1
    #reference[reference > 1] = 1
    #reference[reference < -1] = -1
    return torch.from_numpy(reference)

