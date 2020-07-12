"""
Functions for calculating the utility of color normalisation.

Current implementation provides function for:
1. contrast difference calculation

References:
    .. [1] Mukherjee, J., & Mitra, S. K. (2008). Enhancement of
           color images by scaling the DCT coefficients. IEEE
           Transactions on Image processing, 17(10), 1783-1794.
           :DOI:10.1109/TIP.2008.2002826
    .. [2] Pontalba JT, Gwynne-Timothy T, David E, Jakate K,
           Androutsos D and Khademi A (2019) Assessing the
           Impact of Color Normalization in Convolutional
           Neural Network-Based Nuclei Segmentation Frameworks.
           Front. Bioeng. Biotechnol. 7:300.
           :DOI:10.3389/fbioe.2019.00300
"""

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.util import view_as_blocks
from skimage._shared.utils import check_shape_equality


def contrast_difference(image_true, image_test, window=None,
                        multichannel=False):
    """Determine the contrast difference between two images.

    Parameters
    ----------
    image_true : array-like
        Unnormalised image, in grayscale, RGB or RGBA.
    image_test : array-like
        Normalised image, in grayscale, RGB or RGBA.
    window : tuple, optional
        Window size to compute the contrast difference. Must divide image
        exactly without any padding. Defaults to the entire image size.
    multichannel : bool
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    out : float
        Contrast difference between the test and true images.
        Positive contrast difference indicates test image has higher
        contrast than true image

    References
    ----------
    .. [1] Mukherjee, J., & Mitra, S. K. (2008). Enhancement of
           color images by scaling the DCT coefficients. IEEE
           Transactions on Image processing, 17(10), 1783-1794.
           :DOI:10.1109/TIP.2008.2002826
    .. [2] Pontalba JT, Gwynne-Timothy T, David E, Jakate K,
           Androutsos D and Khademi A (2019) Assessing the
           Impact of Color Normalization in Convolutional
           Neural Network-Based Nuclei Segmentation Frameworks.
           Front. Bioeng. Biotechnol. 7:300.
           :DOI:10.3389/fbioe.2019.00300

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.exposure import adjust_gamma
    >>> image = data.astronaut()
    >>> adjusted_1 = adjust_gamma(image, 2)
    >>> contrast_difference(image, adjusted_1, multichannel=True)
    0.20741130646841788
    >>> adjusted_2 = adjust_gamma(image, 0.5)
    >>> contrast_difference(image, adjusted_2, multichannel=True)
    -0.14646830821582324
    """

    check_shape_equality(image_true, image_test)

    images = [image_test, image_true]
    contrasts = []

    for image in images:
        image = np.asanyarray(image)

        if multichannel:
            if image.shape[-1] not in (3, 4):
                msg = ("The last axis of the input image is interpreted as "
                       "channels. Input image with shape {0} has {1} "
                       "channels in last axis. ``contrast_difference`` is "
                       "implemented for RGB and RGBA images only (if "
                       "mutichannel is true).")
                    raise ValueError(msg.format(image.shape, image.shape[-1]))

            if image.shape[-1] == 4:
                image = rgba2rgb(image)
            if image.shape[-1] == 3:
                image = rgb2gray(image)

        if window is None:
            window = image.shape

        windows = view_as_blocks(image, window)
        new_shape = windows.shape[:(-len(window))] + (-1,)
        flat_windows = windows.reshape(new_shape)

        mean_windows = np.mean(flat_windows, axis=-1)
        std_windows = np.std(flat_windows, axis=-1)

        epsilon = np.finfo(mean_windows.dtype).eps
        mean_windows = np.where(mean_windows == 0, epsilon, mean_windows)
        contrast = std_windows / mean_windows

        contrasts.append(np.mean(contrast))

    return contrasts[0] - contrasts[1]
