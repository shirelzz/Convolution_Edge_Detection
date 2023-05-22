import math
import numpy as np
import cv2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 211551072


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    ker_len = len(k_size)
    sgn_len = len(in_signal)

    assert ker_len <= sgn_len, "Length of the signal is greater than the length of the kernel"

    res_len = ker_len + sgn_len - 1
    result = np.zeros((res_len,))

    for i in range(res_len):
        for j in range(ker_len):
            if 0 <= i - j < sgn_len:
                result[i] += in_signal[i - j] * k_size[j]

    return result
    # return np.convolve(in_signal, k_size, mode='full')


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    assert len(in_image) >= len(kernel), "Length of the image is greater than the length of the kernel"

    img_h, img_w = in_image.shape
    k_size = kernel.shape[0]

    pad = k_size // 2
    padded_img = np.pad(in_image, pad, mode='edge')

    result = np.zeros((img_h, img_w))

    for i in range(img_h):
        for j in range(img_w):
            mat = padded_img[i:i + k_size, j:j + k_size]
            mat_ker_mult = np.sum(mat * kernel).round()
            result[i, j] = mat_ker_mult

    # return result
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """

    # Compute kernels in the x and y directions
    ker_x = np.array([1, 0, -1], dtype=np.float32)
    ker_y = np.array([1, 0, -1], dtype=np.float32).reshape((3, 1))

    # Compute derivatives in the x and y directions
    # derivative_x = cv2.filter2D(in_image, -1, ker_x, borderType=cv2.BORDER_REPLICATE)
    # derivative_y = cv2.filter2D(in_image, -1, ker_y, borderType=cv2.BORDER_REPLICATE)
    derivative_x = conv2D(in_image, ker_x)
    derivative_y = conv2D(in_image, ker_y)

    # Compute magnitude and direction
    direction = np.arctan2(derivative_y, derivative_x)
    magnitude = np.sqrt(derivative_x ** 2 + derivative_y ** 2)

    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create a Gaussian kernel.
    kernel = getGaussianKernel(k_size, float(2))

    blurred_image = conv2D(in_image, kernel)

    return blurred_image


def getGaussianKernel(k_size: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel.
    :param sigma: The standard deviation of the Gaussian distribution.
    :param k_size: Kernel size
    :return: The Gaussian kernel
    """

    assert k_size % 2 == 1, "kernel size has to be an odd number"

    if sigma == 0:
        print("sigma 0")
        sigma = 1e-6

    # kernel = np.zeros(k_size)
    # center = k_size // 2
    # total = 0.0
    #
    # for i in range(k_size):
    #     x = i - center
    #     kernel[i] = np.exp(-0.5 * (x / sigma) ** 2)
    #     total += kernel[i]
    #
    # kernel /= total
    # return kernel

    kernel = np.zeros((k_size, k_size))
    center = k_size // 2
    # total = 0.0
    #
    for x in range(-center, center+1):
        for y in range(-center, center+1):
            x1 = 2 * np.pi * sigma ** 2
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x, y] = 1 / x1 * x2

    # for i in range(k_size):
    #     x = i - center
    #     kernel[i] = np.exp(-0.5 * (x / sigma) ** 2)
    #     total += kernel[i]
    #
    # kernel /= total
    return kernel




def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    kernel = cv2.getGaussianKernel(k_size, -1)
    blurred_image = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    return blurred_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Find zero crossings
    rows, cols = laplacian.shape
    edge_matrix = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [laplacian[i - 1, j], laplacian[i + 1, j], laplacian[i, j - 1], laplacian[i, j + 1]]
            if np.any(np.multiply(neighbors, laplacian[i, j]) < 0):
                edge_matrix[i, j] = 255

    return edge_matrix


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # Apply Laplacian of Gaussian (LoG) operator
    log = cv2.GaussianBlur(img, (5, 5), 0)
    log = cv2.Laplacian(log, cv2.CV_64F)

    # Find zero crossings
    rows, cols = log.shape
    edge_matrix = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [log[i - 1, j], log[i + 1, j], log[i, j - 1], log[i, j + 1]]
            if np.any(np.multiply(neighbors, log[i, j]) < 0):
                edge_matrix[i, j] = 255

    return edge_matrix


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
