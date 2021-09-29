import numpy as np
import skimage.io as sk
from matplotlib import pyplot
from scipy import signal
from pathlib import Path


def vijay_window_square_averaging(input_image, s=5):
    square_filter = np.ones((s, s)) / np.square(s)
    cleaned_image = signal.convolve2d(input_image, square_filter, boundary='symm' , mode='same')

    return cleaned_image


def vijay_error_decreased_unsharp_masking(input_image, blurred_image, test_image):
    size = np.shape(input_image)
    masked_image = input_image - blurred_image
    k = 0.0001
    index_list = []
    error_list = []
    temp = 200
    while temp:
        unsharped_image = input_image + k * masked_image
        error = (1 / (size[0] * size[1])) * np.sum(np.square(test_image - unsharped_image))
        index_list.append(k)
        error_list.append(error)
        k += 0.02
        temp -= 1

    k_min = index_list[error_list.index(min(error_list))]
    print('K value:', k_min)
    print('Error:', error_list[index_list.index(k_min)])
    unsharped_image = input_image + k_min * masked_image

    return [unsharped_image, index_list, error_list]


def vijay_sinusoidal_wave_generation(m, n, u0, v0):
    sinusoidal_image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            degree = np.degrees(((2 * np.pi) / m) * (u0 * i + v0 * j))
            # degree = np.degrees(i+j)
            sinusoidal_image[i, j] = np.sin(degree)

    dft_sinusoidal_image = np.fft.fft2(sinusoidal_image)

    return [sinusoidal_image, dft_sinusoidal_image]


def vijay_idal_lowpass_filter(input_image):
    dft_character_image = np.fft.fftshift(np.fft.fft2(input_image))

    size = np.shape(dft_character_image)
    low_pass_filter = np.zeros(size)
    # d0 = size[0]/10
    d0 = 80
    for i in range(size[0]):
        for j in range(size[1]):
            if np.sqrt(np.square(i - (size[0] / 2)) + np.square(j - (size[1] / 2))) < d0:
                low_pass_filter[i, j] = 1

    dft_filtered_image = dft_character_image * low_pass_filter
    filtered_image = np.fft.ifft2(np.fft.fftshift(dft_filtered_image))

    return [dft_character_image, low_pass_filter, dft_filtered_image, filtered_image]


def vijay_gaussian_filtering(input_image):
    dft_character_image = np.fft.fftshift(np.fft.fft2(input_image))

    size = np.shape(dft_character_image)
    gaussian_filter = np.ones(size)
    # d0 = size[0]/4
    d0 = 100
    for i in range(size[0]):
        for j in range(size[1]):
            gaussian_filter[i, j] = np.exp(
                (-np.square(np.sqrt(np.square(i - (size[0] / 2)) + np.square(j - (size[1] / 2))))) / (
                            2 * np.square(d0)))

    dft_filtered_image = dft_character_image * gaussian_filter
    filtered_image = np.fft.ifft2(np.fft.fftshift(dft_filtered_image))

    return [dft_character_image, gaussian_filter, dft_filtered_image, filtered_image]


def vijay_homomorphic_filter(input_image):
    size = np.shape(input_image)
    pet_image1 = np.ones(size)
    pet_image1 = pet_image1 + input_image

    nonlinear_pet_image = np.log(pet_image1).astype(float)
    dft_nonlinear_pet_image = np.fft.fftshift(np.fft.fft2(nonlinear_pet_image))
    god_filter = np.zeros(size)
    d0 = 80
    yh = 2
    yl = 0.5
    y_diff = yh - yl
    for i in range(size[0]):
        for j in range(size[1]):
            d = np.sqrt(np.square(i - (size[0] / 2)) + np.square(j - (size[1] / 2)))
            god_filter[i, j] = (y_diff * (1 - (np.exp(-np.square(d) / (2 * np.square(d0)))))) + yl
    dft_output_image = np.ones(size)
    dft_output_image = dft_output_image * dft_nonlinear_pet_image * god_filter
    inverse_dft_image = np.fft.ifft2(np.fft.fftshift(dft_output_image))
    output_image = np.exp(inverse_dft_image)

    return [nonlinear_pet_image, dft_nonlinear_pet_image, god_filter, dft_output_image, output_image]


def vijay_mse_laplacian(g_dft, h_dft):
    index_list = []
    error_list = []
    k = -1
    temp = 200
    while temp:
        error = np.sum(np.square(np.absolute((k * h_dft) - g_dft))) / 25
        index_list.append(k)
        error_list.append(error)
        k += 0.01
        temp -= 1

    return [index_list, error_list]
