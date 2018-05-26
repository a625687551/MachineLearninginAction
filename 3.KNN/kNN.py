# coding: utf-8

from numpy import *


def classify0(x, dataset, labels, k):
    data_size = dataset.shape[0]
    mat_diff = tile(x, (data_size, 1)) - dataset
    sq_diff_mat = mat_diff ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_dist = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    matrix = zeros((numberOfLines, 3))
    label_matrix = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromline = line.split("\t")
        matrix[index, :] = listFromline[0:3]
        label_matrix.append(int(listFromline[-1]))
        index += 1
    return matrix, label_matrix


def auto_norm(dataset):
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    ranges = max_val - min_val
    norm_data_set = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_data_set = dataset - tile(min_val, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_val


def img_to_vector(filename):
    img_vector = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                img_vector[0, 32 * j + j] = int(line_str[j])
    return img_vector


def handwriting_test():
    hw_label = []
    error_count = 0
    train_file_list = os.listdir("trainingDigits")
    m = len(train_file_list)
    train_matrix = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_file_list[i]
        file_str = file_name.split(".")[0]
        class_name = int(file_str.split("_")[0])
        hw_label.append(class_name)
        train_matrix[i, :] = img_to_vector("trainingDigits/{}".format(file_name))
    test_file_list = os.listdir("testDigits")
    n = len(test_file_list)
    for i in range(n):
        file_name = test_file_list[i]
        file_str = file_name.split(".")[0]
        class_name = int(file_str.split("_")[0])
        vector_test = img_to_vector("trainingDigits/{}".format(file_name))
        classifer_result = classify0(vector_test, train_matrix, hw_label, 9)
        print("the classifier came back with: {}, the real answer is {}".format(classifer_result, class_name))
        if classifer_result != class_name:
            error_count += 1
    print(" \n the total number of errors is: {}".format(error_count))
    print(" \n the total number of errors rate is: {}".format(error_count / float(n)))
