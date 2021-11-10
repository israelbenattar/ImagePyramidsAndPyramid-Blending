import os

import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.color
from scipy import signal, ndimage


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: The filename of an image on disk.
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
                           image (1) or an RGB image (2).
    :return: This function returns an image.
    """
    image = imageio.imread(filename)
    if len(image.shape) == 3 and representation == 1:
        image = skimage.color.rgb2gray(image)
    if image.max() > 1:
        image = np.divide(image, 255)
    return image.astype(np.float64)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function build the gaussian pyramid of the given image.
    :param im: A grayscale image with double values in [0, 1].
    :param max_levels: The maximal number of levels in the resulting pyramid
    :param filter_size: The size of the Gaussian filter
    :return: The resulting pyramid as a standard python array, the filter vector.
    """
    if filter_size == 1:
        filter_vec = np.array([1]).reshape(1, 1)
    else:
        a = np.array([1, 1]).reshape(1, 2)
        filter_vec = a
        while filter_vec.shape[1] < filter_size:
            filter_vec = signal.convolve2d(a, filter_vec)
        filter_vec = filter_vec / filter_vec.sum()
    pry = [im]
    i = 1
    while i < max_levels and pry[-1].shape[0] > 16 and pry[-1].shape[1] > 16:
        temp = ndimage.filters.convolve(pry[-1], filter_vec)
        temp = ndimage.filters.convolve(temp, np.transpose(filter_vec))

        pry.append(temp[::2, ::2])
        i += 1
    return pry, filter_vec


def _expand(layer, filter_vec):
    a = np.zeros((layer.shape[0] * 2, layer.shape[1] * 2))
    a[::2, ::2] = layer
    a = ndimage.filters.convolve(a, 2 * filter_vec)
    return ndimage.filters.convolve(a, np.transpose(2 * filter_vec))


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function build the laplacian pyramid of the given image.
    :param im: A grayscale image with double values in [0, 1].
    :param max_levels: The maximal number of levels in the resulting pyramid
    :param filter_size: The size of the Gaussian filter
    :return: The resulting pyramid as a standard python array, the filter vector.
    """
    gauss_pry, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pry = []
    for i in range(len(gauss_pry) - 1):
        g_i1_expand = _expand(gauss_pry[i + 1], filter_vec)
        pry.append(np.subtract(gauss_pry[i], g_i1_expand))
    pry.append(gauss_pry[-1])
    return pry, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    This func o implement the reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: The Laplacian pyramid.
    :param filter_vec: The filter vec of the pyramid.
    :param coeff: The corresponding coefficient of the pyramid.
    :return: An image.
    """
    expand_layer = coeff[-1] * lpyr[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        expand = _expand(expand_layer, filter_vec)
        expand_layer = np.add(expand, coeff[i - 1] * lpyr[i - 1])
    return expand_layer


def _stretch_values(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def render_pyramid(pyr, levels):
    """
    :param pyr: Either a Gaussian or Laplacian pyramid.
    :param levels: The number of levels in pyr.
    :return: A single black image in which the pyramid levels of the given
             pyramid pyr are stacked horizontally
    """
    num_of_layer = min(levels, len(pyr))
    render_pyr = pyr[0]
    for i in range(1, num_of_layer):
        current_layer = _stretch_values(np.copy(pyr[i]))
        current_layer.resize(render_pyr.shape[0], current_layer.shape[1])
        render_pyr = np.concatenate((render_pyr, current_layer), axis=1)
    return render_pyr


def display_pyramid(pyr, levels):
    """
    This function display the render pyramid.
    :param pyr: Either a Gaussian or Laplacian pyramid.
    :param levels: The number of levels in pyr.
    """
    rend_pyr = render_pyramid(pyr, levels)
    plt.imshow(rend_pyr)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: The first grayscale images to be blended.
    :param im2: The second Grayscale images to be blended.
    :param mask: A boolean mask containing True and False representing which
                 parts of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: Parameter that used when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: The size of the Gaussian filter which defining the filter used in
                           the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: The size of the Gaussian filter which defining the filter used in
                             the construction of the Gaussian pyramids of the mask.
    :return: The blended image of im1 and im2 using the mask.
    """
    l1, l1_filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, l2_filter = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, gm_filter = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    levels = len(l1)
    im_out_pyr = []
    for i in range(levels):
        im_out_pyr.append(gm[i] * l1[i] + (1 - gm[i]) * l2[i])
    return np.clip(laplacian_to_image(im_out_pyr, l1_filter, [1] * levels), 0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def _example_helper(im1, im2, mask):
    im1 = read_image(relpath(im1), 2)
    im2 = read_image(relpath(im2), 2)
    mask = read_image(relpath(mask), 1)
    mask = np.round(mask).astype(bool)
    r = pyramid_blending(im1[::, ::, 0], im2[::, ::, 0], mask, 3, 5, 5)
    g = pyramid_blending(im1[::, ::, 1], im2[::, ::, 1], mask, 3, 5, 5)
    b = pyramid_blending(im1[::, ::, 2], im2[::, ::, 2], mask, 3, 5, 5)
    bland_image = np.dstack((r, g, b))
    i, j = plt.subplots(2, 2)
    j[0, 0].imshow(im1, cmap='gray')
    j[0, 1].imshow(im2, cmap='gray')
    j[1, 0].imshow(mask, cmap='gray')
    j[1, 1].imshow(bland_image, cmap='gray')
    plt.show()
    return im1, im2, mask, bland_image


def blending_example1():
    im2 = 'two2.jpg'
    im1 = 'one2.jpg'
    mask = 'mask.png'
    return _example_helper(im1, im2, mask)


def blending_example2():
    im1 = 'spongbob.png'
    im2 = 'video-obama-superJumbo_2048x1024.jpg'
    mask = 'spongbob_mask.png'
    return _example_helper(im1, im2, mask)


