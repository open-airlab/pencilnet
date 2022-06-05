import cv2
import numpy as np
import random
import os

from enum import Enum

class KernelType(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class DirectedBlurFilter:
    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, bgr_img):
        return cv2.filter2D(bgr_img, -1, self.kernel)

class KernelBuilder:
    # The greater the size, the more the motion.
    def build_kernel(self, kernel_size = 30, kernel_type = KernelType.HORIZONTAL):
        kernel = np.zeros((kernel_size, kernel_size))

        if kernel_type is KernelType.VERTICAL:    
            kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
            kernel /= kernel_size

        elif kernel_type is KernelType.HORIZONTAL:
            kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
            kernel /= kernel_size
        
        else:
            raise Exception("Ivalid krenel type.")

        return kernel

    def custom_rotated_horizontal_kernel(self, kernel_size, angle_in_degrees = 0, random_angle = False):
        if random_angle:
            angle_in_degrees = random.randint(1, 359)
            print(f'[KernelBuilder] Random angle is {angle_in_degrees} degrees')
        
        (cX, cY) = (kernel_size // 2, kernel_size // 2)
        horizontal_kernel = self.build_kernel(kernel_size=kernel_size)

        M = cv2.getRotationMatrix2D((cX, cY), angle_in_degrees, 1.0)
        rotated = cv2.warpAffine(horizontal_kernel, M, (kernel_size, kernel_size))
        return rotated

if __name__ == '__main__':
    print('Testing kernel geeration...')
    kernel_builder = KernelBuilder()
    kernel_size = 10
    h_kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, angle_in_degrees=0)
    v_kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, angle_in_degrees=90)
    diag_kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, angle_in_degrees=45)
    random_kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, random_angle=True)
    
    kernels = [h_kernel, v_kernel, diag_kernel, random_kernel]

    test_img_path = os.path.join('Public_datasets', 'Test data', 'rgb_real_N_40', 'images','000735.png')
    test_img_bgr = cv2.imread(test_img_path)
    
    imgs = [test_img_bgr]
    for kernel in kernels:
        directed_blur_filter = DirectedBlurFilter(kernel)
        imgs.append(directed_blur_filter.apply(test_img_bgr))

    # Visualize
    # Kernels
    cv2.imshow('h_kernel', np.concatenate(kernels, axis=1))

    # Images
    cv2.imshow('Blured Images', np.concatenate(imgs, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()