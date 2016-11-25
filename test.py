import cv2
import numpy as np

#######   training part    ###############
# samples = np.loadtxt('trained_data/generalsamples.data', np.float32)
# responses = np.loadtxt('trained_data/generalresponses.data', np.float32)

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)

responses = responses.reshape((responses.size, 1))

model = cv2.KNearest()
model.train(samples, responses)


############################# testing part  #########################


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


im = cv2.imread('plates/L1_front.jpg')
# im = cv2.imread('plates/L2_front.png')
# im = cv2.imread('plates/L3_front.png')
# im = cv2.imread('plates/L4_front.png')

plate_height, plate_width = im.shape[:2]
x_min = plate_width / 19
x_max = plate_width / 2.3
y_min = plate_height / 2.4
y_max = plate_height / 1.1
part = im[y_min:y_max, x_min:x_max]

div_factor = 3
if plate_width <= 400:
    div_factor = 1

part = cv2.resize(part, (int(x_max - x_min) / div_factor, int(y_max - y_min) / div_factor))

out = np.zeros(part.shape, np.uint8)
gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 50, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print type(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
(contours, boundingBoxes) = sort_contours(contours)
print type(contours)

string_cumulative = ""

for cnt in contours:
    if cv2.contourArea(cnt) > 50:  # was 50
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            cv2.imshow("roi", roi)
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k=1)
            string = str(int((results[0][0])))
            string_cumulative += string
            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

print string_cumulative
cv2.imshow('im', im)
cv2.imshow('out', out)

cv2.imshow('thresh', thresh)
cv2.imshow('gray', gray)
cv2.waitKey(0)
