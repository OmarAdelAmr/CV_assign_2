#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from connected_component_labeling import question_1_connected_component_labeling_l1
from connected_component_labeling import question_1_connected_component_labeling_l3
from numbers_recognition_test import extract_numbers
from letters_recognition_test import extract_letters


def question1_L1():
    number_of_classes = question_1_connected_component_labeling_l1(cv2.imread("plates/originals/L1.jpg", 0))
    print "Number of classes in L1 binary: ", number_of_classes
    print "------------------------------------------"


def question1_L3():
    print "Please wait..."
    number_of_classes = question_1_connected_component_labeling_l3(cv2.imread("plates/L3_front.png", 0))
    print "Number of classes in L3 gray", number_of_classes
    print "------------------------------------------"


def question2_L1():
    l1_front = cv2.imread('plates/L1_front.png')
    numbers = extract_numbers(l1_front)
    letters = extract_letters(l1_front)
    print "L1 Plate number is:"
    print letters, " ", numbers
    print "------------------------------------------"


def question2_L3():
    l3_front = cv2.imread('plates/L3_front.png')
    numbers = extract_numbers(l3_front)
    letters = extract_letters(l3_front)
    print "L3 Plate number is: "
    print letters, " ", numbers
    print "------------------------------------------"


question1_L1()
question1_L3()
question2_L1()
question2_L3()
