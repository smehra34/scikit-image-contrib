from skimage._shared.testing import assert_equal, assert_almost_equal
from skimage._shared import testing
from skimage.color import rgb2gray, rgba2rgb
import numpy as np

from skimage_contrib.color import contrast_difference


def test_contrast_difference_errors():

    x = np.ones((10, 10))
    # shape mismatch
    with testing.raises(ValueError):
        contrast_difference(x[:-1], x)

    # invalid window
    with testing.raises(ValueError):
        contrast_difference(x, x, window=2)
    with testing.raises(ValueError):
        contrast_difference(x, x, window=(2, 2, 2))
    with testing.raises(ValueError):
        contrast_difference(x, x, window=(2, 3))

    # invalid channels
    with testing.raises(ValueError):
        contrast_difference(x, x, multichannel=True)


def test_contrast_difference_grayscale():

    image1 = np.linspace(0, 0.4, 100)
    image2 = np.linspace(0.4, 0.8, 100)
    image3 = image1 * 2

    assert_equal(contrast_difference(image1, image2),
                 np.std(image2)/np.mean(image2)-np.std(image1)/np.mean(image1))

    assert_equal(contrast_difference(image1, image3), 0)

    assert_equal(contrast_difference(image1, image3),
                 contrast_difference(image1.reshape(10, 10),
                                     image3.reshape((10, 10))))


def test_contrast_difference_multichannel():

    N = 10
    rgb_X = np.random.rand(N, N, N, 3)
    rgb_Y = np.random.rand(N, N, N, 3)

    rgba_X = np.random.rand(N, N, 4)
    rgba_Y = np.random.rand(N, N, 4)

    assert_equal(contrast_difference(rgb_X, rgb_Y, multichannel=True),
                 contrast_difference(rgb2gray(rgb_X), rgb2gray(rgb_Y)))

    assert_equal(contrast_difference(rgba_X, rgba_Y, multichannel=True),
                 contrast_difference(rgba2rgb(rgba_X), rgba2rgb(rgba_Y),
                                     multichannel=True))


def test_contrast_difference_windows():

    N = 10
    X = np.random.rand(N, N)
    Y = np.random.rand(N, N)

    window = (10, 5)

    contrast_X = (np.std(X[:, :5]) / np.mean(X[:, :5]) +
                  np.std(X[:, 5:]) / np.mean(X[:, 5:])) / 2

    contrast_Y = (np.std(Y[:, :5]) / np.mean(Y[:, :5]) +
                  np.std(Y[:, 5:]) / np.mean(Y[:, 5:])) / 2

    assert_equal(contrast_difference(X, Y, window=window),
                 contrast_Y - contrast_X)
