from re import S
import cv2
from enum import Enum
from pencil_filter import PencilFilter
import numpy as np

import os

class SobelDirection(Enum):
    X_AXES = 1
    Y_AXES = 2
    XY_AXES = 3

class SobelEdgeExtractionFilter:
    def __init__(self, sobel_direction = SobelDirection.XY_AXES, kernel_size = 3):
        self.sobel_direction = sobel_direction
        self.kernel_size = kernel_size
    
    def apply(self, bgr_img):
        image_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        if self.sobel_direction is SobelDirection.X_AXES:
            sobel_filtered_image = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize = self.kernel_size)
        elif self.sobel_direction is SobelDirection.Y_AXES:
            sobel_filtered_image = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize = self.kernel_size)
        elif self.sobel_direction is SobelDirection.XY_AXES:
            sobel_filtered_image = cv2.Sobel(image_gray, cv2.CV_64F, 1, 1, ksize = self.kernel_size)
        else:
            raise f'Invalid Sobel direction provided {self.sobel_direction}'
        
        uint_img = np.uint8(sobel_filtered_image)
        return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2RGB)

class CannyEdgeExtractionFilter:
    def __init__(self, lower_threshold = 100, upper_threshold = 200):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
    
    def apply(self, bgr_img):
        image_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.Canny(image_gray, threshold1=self.lower_threshold, threshold2=self.upper_threshold)

        uint_img = np.uint8(filtered_image)
        return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2RGB)

class LaplacianEdgeExtractionFilter:
    def __init__(self, laplacian_kernel_size = 3, blur_kernel_size = 3):
        self.laplacian_kernel_size = laplacian_kernel_size
        self.blur_kernel_size = blur_kernel_size

    
    def apply(self, bgr_img):
        image_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # Reduce noise in image
        img = cv2.GaussianBlur(image_gray,(self.blur_kernel_size, self.blur_kernel_size),0)

        filtered_image = cv2.Laplacian(img, ksize=self.laplacian_kernel_size, ddepth=cv2.CV_16S)
        # converting back to uint8
        filtered_image = cv2.convertScaleAbs(filtered_image)
        
        uint_img = np.uint8(filtered_image)
        return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    # Test filters
    test_img_path = os.path.join('Public_datasets', 'Test data', 'rgb_real_N_40', 'images','000735.png')
    test_img_bgr = cv2.imread(test_img_path)

    pencil_filter = PencilFilter()

    sobel_filter_x = SobelEdgeExtractionFilter(sobel_direction=SobelDirection.X_AXES)
    sobel_filter_y = SobelEdgeExtractionFilter(sobel_direction=SobelDirection.Y_AXES)
    sobel_filter_xy = SobelEdgeExtractionFilter()

    canny_filter = CannyEdgeExtractionFilter()
    canny_filter_10 = CannyEdgeExtractionFilter(lower_threshold=10)

    laplacian_filter = LaplacianEdgeExtractionFilter()

    filters = [pencil_filter, sobel_filter_x, sobel_filter_y, sobel_filter_xy, canny_filter, canny_filter_10, laplacian_filter]

    imgs = [test_img_bgr]
    for f in filters:
        imgs.append(f.apply(test_img_bgr))
    
    cv2.imshow('Filtered Images', np.concatenate(imgs, axis=1))
    cv2.imwrite('FilteredImages.png', np.concatenate(imgs, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()