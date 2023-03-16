import numpy as np
import math


def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """
    attributes = D[0]
    classes = D[1]
    H_D = calculate_entropy(classes)
    attributes_on_split_index = attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, classes, value)
    H_Dy = calculate_entropy(np.array(class_y))
    H_Dn = calculate_entropy(np.array(class_n))
    average_entropy = (len(class_y)/classes.size * H_Dy) + (len(class_n)/classes.size * H_Dn)
    information_gain = H_D - average_entropy
    return information_gain


def split_into_classes(attribute_to_split, classes, value):
    class_y = []
    class_n = []
    for num in range(np.size(attribute_to_split)):
        if attribute_to_split[num] <= value:
            class_y.append(classes[num])
        else:
            class_n.append(classes[num])
    return class_y, class_n


def calculate_entropy(class_):
    """
    A function that calculates the entropy
    Args:
        class_
    Returns:
        a float value of the entropy
    """
    nn = class_.tolist()
    n = len(nn)
    n_0 = nn.count(0)
    n_1 = nn.count(1)
    if n == n_0 or n == n_1 or n == 0:
        return 0
    elif n_0 == n_1:
        return 1
    else:
        p_c0_D = n_0 / n
        p_c1_D = n_1 / n
        h_d_0 = p_c0_D * math.log2(p_c0_D)
        h_d_1 = p_c1_D * math.log2(p_c1_D)
        H_D = -1 * (h_d_0 + h_d_1)
        return H_D


def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """
    attributes = D[0]
    classes = D[1]
    attributes_on_split_index = attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, classes, value)
    G_Dy = calculate_gini_index(np.array(class_y))
    G_Dn = calculate_gini_index(np.array(class_n))
    average_gini = (len(class_y)/classes.size * G_Dy) + (len(class_n)/classes.size * G_Dn)
    return average_gini


def calculate_gini_index(class_):
    nn = class_.tolist()
    n = len(nn)
    n_0 = nn.count(0)
    n_1 = nn.count(1)
    if n == 0:
        return 0
    else:
        p_c0_D_square = (n_0 / n)**2
        p_c1_D_square = (n_1 / n)**2
        gini = 1 - (p_c1_D_square + p_c0_D_square)
        return gini


def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """
    attributes = D[0]
    classes = D[1]
    attributes_on_split_index = attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, classes, value)
    pre = 2 * (len(class_y)/classes.size) * (len(class_n)/classes.size)
    if len(class_n) == 0:
        p_c0_Dn = p_c1_Dn = 0
    else:
        p_c0_Dn = class_n.count(0) / len(class_n)
        p_c1_Dn = class_n.count(1) / len(class_n)
    if len(class_y) == 0:
        p_c0_Dy = p_c1_Dy = 0
    else:
        p_c0_Dy = class_y.count(0) / len(class_y)
        p_c1_Dy = class_y.count(1) / len(class_y)
    p_c0_Dy_minus_p_c0_Dn_abs = abs(p_c0_Dy - p_c0_Dn)
    p_c1_Dy_minus_p_c1_Dn_abs = abs(p_c1_Dy - p_c1_Dn)
    summation = p_c1_Dy_minus_p_c1_Dn_abs + p_c0_Dy_minus_p_c0_Dn_abs
    cart = pre * summation
    return cart


def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes'
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """
    attributes = D[0]
    best_index = 0
    best_value = 0
    best_ig = -np.inf
    best_gini = np.inf
    best_cart = -np.inf
    for index in range(len(attributes[1])):
        values = attributes[index]
        unique_values = np.unique(values)
        for value in unique_values:
            if criterion == "IG":
                information_gain = IG(D, index, value)
                if information_gain > best_ig:
                    best_ig = information_gain
                    best_index = index
                    best_value = value
            elif criterion == "GINI":
                gini_index = G(D, index, value)
                if gini_index < best_gini:
                    best_gini = gini_index
                    best_index = index
                    best_value = value
            elif criterion == "CART":
                cart = CART(D, index, value)
                if cart > best_cart:
                    best_cart = cart
                    best_index = index
                    best_value = value
            else:
                print("criterion must be one of 'IG', 'GINI', or 'CART'")
    print(f"Best split for {criterion}(index, value) = ({best_index}, {best_value})")
    return best_index, best_value

# functions are first class objects in python, so let's refer to our desired criterion by a single name


def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of
        the classes of the observations, in the same order
    """
    data = np.loadtxt(filename, delimiter=',')
    size_attribute = len(data[1])-1
    attributes = data[:, 0:size_attribute]
    classes = data[:, -1]


    return attributes, classes


def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    train_attributes = train[0]
    train_class = train[1]
    (index, value) = bestSplit(train, "IG")

    attributes_on_split_index = train_attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, train_class, value)
    if class_y.count(0) > class_y.count(1):
        label_y = 0
    else:
        label_y = 1
    if class_n.count(0) > class_n.count(1):
        label_n = 0
    else:
        label_n = 1

    test_attributes = test[0]
    attributes_test_index = test_attributes[:, index]
    test_class_predicted = split_into_classes_and_predict(attributes_test_index, value, label_y, label_n)
    print(f'predicted on IG: {test_class_predicted}')



def split_into_classes_and_predict(attribute_to_split, value, y_label, n_label):
    test_class_predicted = []
    for num in range(np.size(attribute_to_split)):
        if attribute_to_split[num] <= value:
            test_class_predicted.append(y_label)
        else:
            test_class_predicted.append(n_label)
    return test_class_predicted


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    train_attributes = train[0]
    train_class = train[1]
    (index, value) = bestSplit(train, "GINI")

    attributes_on_split_index = train_attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, train_class, value)
    if class_y.count(0) > class_y.count(1):
        label_y = 0
    else:
        label_y = 1
    if class_n.count(0) > class_n.count(1):
        label_n = 0
    else:
        label_n = 1

    test_attributes = test[0]
    attributes_test_index = test_attributes[:, index]
    test_class_predicted = split_into_classes_and_predict(attributes_test_index, value, label_y, label_n)
    print(f'predicted on GINI: {test_class_predicted}')


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """
    train_attributes = train[0]
    train_class = train[1]
    (index, value) = bestSplit(train, "CART")

    attributes_on_split_index = train_attributes[:, index]
    class_y, class_n = split_into_classes(attributes_on_split_index, train_class, value)
    if class_y.count(0) > class_y.count(1):
        label_y = 0
    else:
        label_y = 1
    if class_n.count(0) > class_n.count(1):
        label_n = 0
    else:
        label_n = 1

    test_attributes = test[0]
    attributes_test_index = test_attributes[:, index]
    test_class_predicted = split_into_classes_and_predict(attributes_test_index, value, label_y, label_n)
    print(f'predicted on CART: {test_class_predicted}')


def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    # index = 0
    # value = 0
    train = load('./train.txt')
    test = load('./test.txt')
    classifyIG(train, test)
    print()
    classifyG(train, test)
    print()
    classifyCART(train, test)


if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    main()
