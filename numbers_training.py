import sys

import numpy as np
import cv2

im = cv2.imread('numbers_training_set/numbers_set.png')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
threshold = cv2.adaptiveThreshold(blur_gaussian, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100))
responses = []
keys = [i for i in range(48, 58)]

for contour in contours:
    if cv2.contourArea(contour) > 50:
        [x, y, w, h] = cv2.boundingRect(contour)

        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            char_im = threshold[y:y + h, x:x + w]
            char_im_small = cv2.resize(char_im, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = char_im_small.reshape((1, 100))
                samples = np.append(samples, sample, 0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print "training complete"

np.savetxt('generalsamplesNumbers2.data', samples)  # generalsamplesNumbers is used
np.savetxt('generalresponsesNumbers2.data', responses)
