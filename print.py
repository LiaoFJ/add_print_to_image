import cv2
import numpy as np
import os
path = './'

def calc_diff(pixel, bg_color):
    sum = 0
    for i in range(3):
        if pixel[i] >= bg_color[i]:
            sum += (pixel[i]-bg_color[i])**2
        else:
            sum += (bg_color[i]-pixel[i])**2
    return sum

def horizontal_flip(image):
    # 50% flipping image, axis 0 vertical, 1 horizontal
    flip_prop = np.random.randint(low=0, high=2)
    axis =np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)
    return image

def crop_image(image, image_size):
    x_offset = np.random.randint(low=0, high=image_size)
    y_offset = np.random.randint(low=0, high=image_size)
    image = image[x_offset: x_offset + image_size, y_offset: y_offset + image_size, :]
    return image

def get_print_to_image(pattern, image, image_size, pattern_size, threshold, if_extra, if_real_img):
    #get image
    get_image = cv2.imread(os.path.join(path, image))
    dim_image = (image_size, image_size)
    resized = cv2.resize(get_image, dim_image, interpolation=cv2.INTER_AREA)

    # #get gray from image
    #
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # #get x-gradient and y-gradient
    # gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    # gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    #
    # # subtract the y-gradient from the x-gradient
    #
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # blurred = cv2.blur(gradient, (3, 3))
    #
    # # get closed
    #
    # (_, thresh) = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # closed = cv2.erode(closed, kernel_2)

    #another way to get closed
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(gray, (3, 3), 0)
    Edge = cv2.Canny(Blur, 10, 200)
    Edge = cv2.dilate(Edge, None)
    Edge = cv2.erode(Edge, None)
    Contour, _ = cv2.findContours(Edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contour = sorted(Contour, key=cv2.contourArea, reverse=True)
    Max_contour = Contour[0]
    Epsilon = 0.001 * cv2.arcLength(Max_contour, True)
    Approx = cv2.approxPolyDP(Max_contour, Epsilon, True)
    Mask = np.zeros(resized.shape[:2], np.uint8)
    cv2.drawContours(Mask, [Approx], -1, 255, -1)
    closed = cv2.GaussianBlur(Mask, (5, 5), 0)

    #get pattern
    get_pattern = cv2.imread(os.path.join(path, pattern))

    #translate tile RGB to LAB so that easy to get back-ground color and    /
    #set background color as the first pixel (may be changed)
    bg_color = cv2.cvtColor(get_pattern, cv2.COLOR_BGR2LAB)[0][0]

    if if_extra:
        dim_pattern = (image_size*2, image_size*2)
        resize_pattern = cv2.resize(get_pattern, dim_pattern, interpolation=cv2.INTER_AREA)
        tile = resize_pattern
        #do flip and random crop
        tile = horizontal_flip(tile)
        tile = crop_image(tile, image_size)
    else:
        dim_pattern = (pattern_size, pattern_size)
        resize_pattern = cv2.resize(get_pattern, dim_pattern, interpolation=cv2.INTER_AREA)
        #expand pattern
        tile = np.tile(resize_pattern, (int(image_size/pattern_size) * 2, int(image_size/pattern_size)* 2, 1))
        #do flip and random crop
        tile = horizontal_flip(tile)
        tile = crop_image(tile, image_size)

    #get print_tile and translate it to LAB channel
    print_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)

    #set threshold
    diff_threshold = threshold

    #do printing
    rows, cols = closed.shape

    # do filtering if the img is real img:
    if if_real_img:
        for i in range(image_size):
            for j in range(image_size):
                if gray[i][j] >= 230:
                    gray[i][j] = 0
    else:
        print('image is just technical drawing')

    if if_real_img:
        for i in range(rows):
            for j in range(cols):
                if calc_diff(print_tile[i][j], bg_color) > diff_threshold and gray[i][j] != 0:
                    #gray-scale
                    resized[i][j][0] = tile[i][j][0] * (gray[i][j]/255)
                    resized[i][j][1] = tile[i][j][1] * (gray[i][j]/255)
                    resized[i][j][2] = tile[i][j][2] * (gray[i][j]/255)
    else:
        for i in range(rows):
            for j in range(cols):
                if calc_diff(print_tile[i][j], bg_color) > diff_threshold and closed[i][j] != 0:
                    #gray-scale
                    resized[i][j][0] = tile[i][j][0] * (gray[i][j]/255)
                    resized[i][j][1] = tile[i][j][1] * (gray[i][j]/255)
                    resized[i][j][2] = tile[i][j][2] * (gray[i][j]/255)

    cv2.imshow('resized', resized)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('result.jpg', resized)
    return resized

if __name__ == '__main__':
    pattern = 'test_floral.jpg'
    image = 'FashionGAN_Result_127.png'
    image_size = 256
    pattern_size = 128
    threshold = 1000
    if_extra = False
    modified_image = get_print_to_image(pattern, image, image_size, pattern_size, threshold, if_extra=False, if_real_img=True)