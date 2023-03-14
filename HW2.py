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
    # Original entropy of the system (H(D)) = - sum(p(c1|D)log2p(c1|D))
    attributes = D[0]
    classes = D[1]
    H_D = calculate_entropy(classes)
    print(f"Initial Entropy H(D) = {H_D}")

    attributes_on_split_index = attributes[:, index]
    class_y = []
    class_n = []
    for num in range(np.size(attributes_on_split_index)):
        if attributes_on_split_index[num] <= value:
            class_y.append(classes[num])
        else:
            class_n.append(classes[num])

    H_Dy = calculate_entropy(np.array(class_y))
    H_Dn = calculate_entropy(np.array(class_n))
    print(f"H(Dy) = {H_Dy}")
    print(f"H(Dn) = {H_Dn}")

    average_entropy = (len(class_y)/classes.size * H_Dy) + (len(class_n)/classes.size * H_Dn)
    print(f"average entropy = {average_entropy}")

    information_gain = H_D - average_entropy
    print(f"Information Gain = {information_gain}")
    return information_gain


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
    p_c0_D = n_0 / n
    p_c1_D = n_1 / n
    if n == n_0 or n == n_1:
        return 0
    elif n_0 == n_1:
        return 1
    else:
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
    G_D = calculate_gini_index(classes)
    print(f"initial gini index G(D) = {G_D}")

    attributes_on_split_index = attributes[:, index]
    class_y = []
    class_n = []
    for num in range(np.size(attributes_on_split_index)):
        if attributes_on_split_index[num] <= value:
            class_y.append(classes[num])
        else:
            class_n.append(classes[num])

    G_Dy = calculate_gini_index(np.array(class_y))
    G_Dn = calculate_gini_index(np.array(class_n))
    print(f"G(Dy) = {G_Dy}")
    print(f"G(Dn) = {G_Dn}")

    average_gini = (len(class_y)/classes.size * G_Dy) + (len(class_n)/classes.size * G_Dn)
    print(f"average gini = {average_gini}")

    gini_index_value = G_D - average_gini
    print(f"gini index value = {gini_index_value}")
    return gini_index_value


def calculate_gini_index(class_):
    nn = class_.tolist()
    n = len(nn)
    n_0 = nn.count(0)
    n_1 = nn.count(1)
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
    class_y = []
    class_n = []
    for num in range(np.size(attributes_on_split_index)):
        if attributes_on_split_index[num] <= value:
            class_y.append(classes[num])
        else:
            class_n.append(classes[num])
    pre = 2 * (len(class_y)/classes.size) * (len(class_n)/classes.size)
    p_c0_Dy = class_y.count(0)/len(class_y)
    p_c0_Dn = class_n.count(0)/len(class_n)
    p_c1_Dy = class_y.count(1) / len(class_y)
    p_c1_Dn = class_n.count(1) / len(class_n)
    p_c0_Dy_minus_p_c0_Dn_abs = abs(p_c0_Dy - p_c0_Dn)
    p_c1_Dy_minus_p_c1_Dn_abs = abs(p_c1_Dy - p_c1_Dn)
    summation = p_c1_Dy_minus_p_c1_Dn_abs + p_c0_Dy_minus_p_c0_Dn_abs
    cart = pre * summation
    print(f"CART(Dy, Dn) = {cart}")
    return cart


def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """


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
    attributes = data[:, 0:10]
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


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    D = load('./test.txt')
    index = 0
    value = 0
    information_gain = IG(D, index, value)
    print()
    print(G(D, index, value))
    print()
    CART(D, index, value)


if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    main()
