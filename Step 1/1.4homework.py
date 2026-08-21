import numpy as np

class ImageProcessor:
    def __init__(self, matrix):
        self._img = matrix

    def get_central_crop(self):
        return self._img[len(self._img)//2-1:len(self._img)//2+1, self._img.shape[1]//2-1:self._img.shape[1]//2+1]
    
    def get_corners(self):
        return np.array([[self._img[0, 0], self._img[0, -1]],
                          [self._img[-1, 0], self._img[-1, -1]]])
    
    def draw_border(self, value):
        self._img[0, :] = value
        self._img[-1, :] = value
        self._img[:, 0] = value
        self._img[:, -1] = value
    
    def filter_corrupted_pixels(self, threshold):
        self._img[self._img > threshold] = np.mean(self._img)

    def get_custom_grid(self, rows_list, cols_list):
        return self._img[np.ix_(rows_list, cols_list)]


if __name__ == "__main__":
    test_img = np.array([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ])

    proc = ImageProcessor(test_img)
    
    print(proc.get_central_crop())

    print(proc.get_corners())

    img = np.array([
        [10, 2,  3,  20],
        [4,  5,  6,  7],
        [8,  9,  1,  2]
    ], dtype = float)

    proc = ImageProcessor(img)

    print(" ")
    print(proc.get_custom_grid([0, 2], [1, 3]))

    proc.filter_corrupted_pixels(5)

    print(" ")
    print(proc._img)
