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

    ker_len = len(kernel)
    img_len = len(in_image)

    assert img_len >= ker_len, "Length of the image is greater than the length of the kernel"

    ker_col_len, ker_row_len = kernel.shape
    img_col_len, img_row_len = in_image.shape
    print("in_image shape:", in_image.shape)
    print("kernel shape:", kernel.shape)

    res_col_len = img_row_len - ker_row_len + 1
    res_row_len = img_col_len - ker_col_len + 1
    print(res_row_len, res_col_len)

    result = np.zeros((res_row_len, res_col_len))

    # pad_col_len = ker_row_len // 2
    # pad_row_len = ker_col_len // 2
    # padded_image = np.pad(in_image, ((pad_row_len, pad_row_len), (pad_col_len, pad_col_len)), mode='edge')

    pad_top = ker_col_len // 2
    pad_bottom = ker_col_len - pad_top - 1
    pad_left = ker_row_len // 2
    pad_right = ker_row_len - pad_left - 1

    padded_image = np.pad(in_image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

    # pad_col_len = img_row_len - res_col_len
    # pad_row_len = img_col_len - res_row_len
    # result = np.pad(result, ((0, pad_row_len), (0, pad_col_len)), mode='constant')

    for i in range(res_row_len):
        for j in range(res_col_len):
            for r in range(ker_row_len):
                for c in range(ker_col_len):
                    if i + r < img_row_len and j + c < img_col_len:
                        result[i][j] += padded_image[i+r][j+c] * kernel[r][c]

    pad_col_len = img_row_len - res_col_len
    pad_row_len = img_col_len - res_row_len
    result = np.pad(result, ((0, pad_col_len), (0, pad_row_len)), mode='edge')

    return result
    # return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


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
    derivative_x = cv2.filter2D(in_image, -1, ker_x, borderType=cv2.BORDER_REPLICATE)
    derivative_y = cv2.filter2D(in_image, -1, ker_y, borderType=cv2.BORDER_REPLICATE)

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

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


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
