import cv2
import numpy as np

# Reading trained data files
samples = np.loadtxt('trained_data_numbers/generalsamplesNumbers.data', np.float32)
responses = np.loadtxt('trained_data_numbers/generalresponsesNumbers.data', np.float32)

responses = responses.reshape((responses.size, 1))

model = cv2.KNearest()
model.train(samples, responses)


# Extracting information


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def extract_numbers(input_image):
    # im = cv2.imread('plates/L1_front.png')
    # im = cv2.imread('plates/L3_front.png')

    im = input_image

    im = cv2.resize(im, (1050, 580))
    plate_height, plate_width = im.shape[:2]

    x_min = plate_width / 19
    x_max = plate_width / 2.3
    y_min = plate_height / 2.4
    y_max = y_min + 255
    # y_max = plate_height / 1.1
    part = im[y_min:y_max, x_min:x_max]

    div_factor = 2

    part = cv2.resize(part, (int((x_max - x_min) / div_factor), int((y_max - y_min) / div_factor)))

    out = np.zeros(part.shape, np.uint8)
    gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 50, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    (contours, boundingBoxes) = sort_contours(contours)
    string_cumulative = ""

    for contour in contours:
        # print cv2.arcLength(cnt, True)
        if cv2.contourArea(contour) > 800:  # was 50

            [x, y, w, h] = cv2.boundingRect(contour)
            if h > 28:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                char_im = threshold[y:y + h, x:x + w]
                char_im_small = cv2.resize(char_im, (10, 10))
                char_im_small = char_im_small.reshape((1, 100))
                char_im_small = np.float32(char_im_small)
                retval, results, neigh_resp, dists = model.find_nearest(char_im_small, k=1)
                string = str(int((results[0][0])))
                # if string == '6' and roi.shape[0] < 100:
                #     string = '9'
                # elif string == '9' and roi.shape[0] > 100:
                #     string = '6'
                string_cumulative += string
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

    return string_cumulative

# print extract_numbers(cv2.imread('plates/L3_front.png'))
# print "Plate No.", string_cumulative
# # cv2.imshow('im', im)
# cv2.imshow('out', out)
# # cv2.imshow('thresh', thresh)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
