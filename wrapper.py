import numpy as np
import skimage.io as sk
from matplotlib import pyplot
from scipy import signal
from pathlib import Path
from allfunctions import *


def problem1_a():
    noisy_image_path = Path('noisy.tif')
    noisy_image = sk.imread(noisy_image_path)
    cleaned_image1 = vijay_window_square_averaging(noisy_image, s=5)
    cleaned_image2 = vijay_window_square_averaging(noisy_image, s=10)
    cleaned_image3 = vijay_window_square_averaging(noisy_image, s=15)
    pyplot.subplot(221)
    pyplot.imshow(noisy_image, cmap='gray')
    pyplot.title('Original Image')
    pyplot.subplot(222)
    pyplot.imshow(cleaned_image1, cmap='gray')
    pyplot.title('Window size 5')
    pyplot.subplot(223)
    pyplot.imshow(cleaned_image2, cmap='gray')
    pyplot.title('Window size 10')
    pyplot.subplot(224)
    pyplot.imshow(cleaned_image3, cmap='gray')
    pyplot.title('Window size 15')
    pyplot.show()


def problem1_b():
    noisy_image_path = Path('noisy.tif')
    character_image_path = Path('characters.tif')
    character_image = sk.imread(character_image_path)
    noisy_image = sk.imread(noisy_image_path)
    cleaned_image = vijay_window_square_averaging(noisy_image, s=15)
    blurred_image = vijay_window_square_averaging(cleaned_image, s=15)
    unsharped_image, index_list, error_list = vijay_error_decreased_unsharp_masking(cleaned_image, blurred_image, character_image)
    pyplot.subplot(221)
    pyplot.imshow(noisy_image, cmap='gray')
    pyplot.subplot(222)
    pyplot.imshow(cleaned_image, cmap='gray')
    pyplot.subplot(223)
    pyplot.plot(index_list, error_list)
    pyplot.subplot(224)
    pyplot.imshow(unsharped_image, cmap='gray')
    pyplot.show()


def problem2_a():
    m, n = 1001, 1001
    u0 = 100
    v0 = 200
    sinusoidal_image, dft_sinusoidal_image = vijay_sinusoidal_wave_generation(m, n, u0, v0)
    pyplot.subplot(122)
    pyplot.imshow(np.log(np.absolute(dft_sinusoidal_image) + 1), cmap='gray')
    pyplot.subplot(121)
    pyplot.imshow(sinusoidal_image, cmap='gray')
    pyplot.show()


def problem2_b():
    character_image_path = Path('noisy.tif')
    character_image = sk.imread(character_image_path)
    dft_character_image, low_pass_filter, dft_filtered_image, filtered_image = vijay_idal_lowpass_filter(character_image)
    pyplot.subplot(231)
    pyplot.imshow(character_image, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(232)
    pyplot.imshow(np.log(np.absolute(dft_character_image) + 1))
    pyplot.title('DFT of Original')
    pyplot.subplot(233)
    pyplot.imshow(low_pass_filter, cmap='gray')
    pyplot.title('Ideal Low pass Filter')
    pyplot.subplot(234)
    pyplot.imshow(np.log(np.absolute(dft_filtered_image) + 1))
    pyplot.title('DFT of lowpassed Image')
    pyplot.subplot(235)
    pyplot.imshow(np.absolute(filtered_image), cmap='gray')
    pyplot.title('Filetered Image')
    pyplot.show()


def problem2_c():
    character_image_path = Path('noisy.tif')
    character_image = sk.imread(character_image_path)
    dft_character_image, low_pass_filter, dft_filtered_image, filtered_image = vijay_gaussian_filtering(character_image)
    pyplot.subplot(231)
    pyplot.imshow(character_image, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(232)
    pyplot.imshow(np.log(np.absolute(dft_character_image) + 1))
    pyplot.title('DFT of Original')
    pyplot.subplot(233)
    pyplot.imshow(low_pass_filter, cmap='gray')
    pyplot.title('Gaussian Filter')
    pyplot.subplot(234)
    pyplot.imshow(np.log(np.absolute(dft_filtered_image) + 1))
    pyplot.title('DFT of lowpassed Image')
    pyplot.subplot(235)
    pyplot.imshow(np.absolute(filtered_image), cmap='gray')
    pyplot.title('Filtered Image')
    pyplot.show()


def problem3():
    pet_image_path = Path('PET_image.tif')
    pet_image = sk.imread(pet_image_path)
    nonlinear_pet_image, dft_nonlinear_pet_image, god_filter, dft_output_image, output_image = vijay_homomorphic_filter(pet_image)
    pyplot.subplot(161)
    pyplot.imshow(pet_image, cmap='gray')
    pyplot.subplot(162)
    pyplot.imshow(nonlinear_pet_image, cmap='gray')
    pyplot.subplot(163)
    pyplot.imshow(np.log(np.absolute(dft_nonlinear_pet_image) + 1))
    pyplot.subplot(164)
    pyplot.imshow(god_filter)
    pyplot.subplot(165)
    pyplot.imshow(np.log(np.absolute(dft_output_image) + 1))
    pyplot.subplot(166)
    pyplot.imshow(np.absolute(output_image), cmap='gray')
    pyplot.show()

def problem4():
    g_time1 = np.reshape([0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, -16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0], (5, 5))
    g_time2 = np.ones((5, 5))
    g_time2[2, 2] = -24

    g_dft1 = np.fft.fft2(g_time1)
    g_dft2 = np.fft.fft2(g_time2)

    h_dft = np.zeros((5, 5))
    for i in range(-2, 3):
        for j in range(-2, 3):
            h_dft[i + 2, j + 2] = (i * i) + (j * j)

    index_list1, error_list1 = vijay_mse_laplacian(g_dft1, h_dft)
    index_list2, error_list2 = vijay_mse_laplacian(g_dft2, h_dft)

    k_min1 = index_list1[error_list1.index(min(error_list1))]
    k_min2 = index_list2[error_list2.index(min(error_list2))]
    print('Filter 1 K value:', k_min1)
    print('Filter 1 Error:', min(error_list1))
    print('Filter 2 K value:', k_min2)
    print('Filter 2 Error:', min(error_list2))
    pyplot.subplot(121)
    pyplot.plot(index_list1, error_list1)
    pyplot.subplot(122)
    pyplot.plot(index_list2, error_list2)
    pyplot.show()


if __name__ == '__main__':
    # problem1_a()
    # problem1_b()
    # problem2_a()
    # problem2_b()
    # problem2_c()
    # problem3()
    # problem4()
