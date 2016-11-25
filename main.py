import cv2 as cv
import numpy as np
import sys


def make_labels(image):
    image_height, image_width = image.shape[:2]
    result = np.zeros((image_height, image_width), np.uint8)
    # result[0][0] = 1
    labels = []
    label_counter = 1
    for y in range(0, image_height):
        for x in range(0, image_width):
            # jjj = raw_input("Press Enter")
            has_similar_label = False
            neighbours = [(y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x - 1)]
            related_labels = ()
            for cell in neighbours:
                if cell[1] < image_width and cell[1] >= 0 and cell[0] < image_height and cell[0] >= 0:
                    # print cell
                    # print y, ", ", x
                    # print labels
                    if image[y][x] == image[cell[0]][cell[1]]:
                        has_similar_label = True
                        related_labels += (result[cell[0]][cell[1]],)
                        result[y][x] = result[cell[0]][cell[1]]

                        # if has_similar_label:
                        #     replacement_index_counter = 0
                        #     for temp_label in labels:
                        #         for lbl in related_labels:
                        #             if temp_label.__contains__(lbl):
                        #                 labels[replacement_index_counter] = tuple(set(lbl + related_labels))
                        #         replacement_index_counter += 1
                        # else:
                        #     result[y][x] = label_counter
                        #     labels.append((label_counter,))
                        #     label_counter += 1
            # print related_labels
            # print labels
            # if len(related_labels) > 1:
            #     print ">1"

            if has_similar_label:
                labels.append(related_labels)
            else:
                result[y][x] = label_counter
                labels.append((label_counter,))
                label_counter += 1

    labels = list(set(labels))
    labels = [set(h) for h in labels if h]
    sys.setrecursionlimit(5000)
    labels = find_intersection(labels)
    return len(labels)


def find_intersection(m_list):
    m_list_duplicate = m_list
    for i, v in enumerate(m_list):
        for j, k in enumerate(m_list[i + 1:], i + 1):
            if v & k:
                m_list_duplicate[i] = v.union(m_list.pop(j))
                return find_intersection(m_list)
    return m_list


def question_1_connected_component_labeling_l1():
    l1_binary_image = cv.imread("L1_binary.jpg", 0)

    result = make_labels(image=l1_binary_image)
    print "Number of classes in L1 Binary = ", result
    return result


def question_1_connected_component_labeling_l3():
    l3_gray_image = cv.imread("L3_gray_front.jpg", 0)
    image_height, image_width = l3_gray_image.shape[:2]
    init_class_image = np.zeros((image_height, image_width), np.uint8)

    for y in range(0, image_height):
        for x in range(0, image_width):
            if l3_gray_image[y][x] >= 0 and l3_gray_image[y][x] < 64:
                init_class_image[y][x] = 1
            elif l3_gray_image[y][x] >= 64 and l3_gray_image[y][x] < 128:
                init_class_image[y][x] = 2
            elif l3_gray_image[y][x] >= 128 and l3_gray_image[y][x] < 192:
                init_class_image[y][x] = 3
            else:
                init_class_image[y][x] = 4

    result = make_labels(image=init_class_image)
    print "Number of classes in L3 Gray = ", result
    return result


# question_1_connected_component_labeling_l1()
# question_1_connected_component_labeling_l3()
# L = [(1, 2, 3, 8), (4,), (1, 3), (2, 3), (4, 5, 8), (5, 6,), (6, 7,)]
# s = [set(i) for i in L if i]
# print find_intersection(s)
# try1()
question_1_connected_component_labeling_l1()
