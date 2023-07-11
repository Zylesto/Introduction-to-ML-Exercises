import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC


def create_database_from_folder(path):
    '''
    DON'T CHANGE THIS METHOD.
    If you run the Online Detection, this function will load and reshape the
    images located in the folder. You pass the path of the images and the function returns the labels,
    training data and number of images in the database
    :param path: path of the training images
    :return: labels, training images, number of images
    '''
    labels = list()
    filenames = np.sort(glob.glob(path))
    num_images = len(filenames)
    train = np.zeros((N * N, num_images))
    for n in range(num_images):
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (N, N))
        assert img.shape == (N, N), 'Image {0} of wrong size'.format(filenames[n])
        train[:, n] = img.reshape((N * N))
        labels.append(filenames[n].split("eigenfaces/")[1].split("_")[0])
    print('Database contains {0} images'.format(num_images))
    labels = np.asarray(labels)
    return labels, train, num_images


def process_and_train(labels, train, num_images, h, w):
    '''
    Calculate the essentials: the average face image and the eigenfaces.
    Train the classifier on the eigenfaces and the given training labels.
    :param labels: 1D-array
    :param train: training face images, 2D-array with images as row vectors (e.g. 64x64 image ->  4096 vector)
    :param num_images: number of images, int
    :param h: height of an image
    :param w: width of an image
    :return: the eigenfaces as row vectors (2D-array), number of eigenfaces, the average face
    '''
    avg = calculate_average_face(train)
    eigenfaces, num_eigenfaces = calculate_eigenfaces(train, avg, num_images, h, w)
    features = get_feature_representation(train.T, eigenfaces, avg, num_eigenfaces)

    # Train the classifier using the calculated features
    clf = SVC(kernel="linear", C=0.025)
    clf.fit(features, labels)

    return eigenfaces, num_eigenfaces, avg


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''
    return np.mean(train, axis=1)


def calculate_eigenfaces(train, avg, num_eigenfaces, h, w):
    '''
    Calculate the eigenfaces from the given training set using SVD
    :param train: training face images, 2D-array with images as row vectors
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to return from the computed SVD
    :param h: height of an image in the training set
    :param w: width of an image in the training set
    :return: the eigenfaces as row vectors, 2D-array --> shape(num_eigenfaces, #pixel of an image)
    '''
    centered_train = train - avg[:, np.newaxis]
    _, s, vh = np.linalg.svd(centered_train, full_matrices=False)
    eigenfaces = vh[:num_eigenfaces, :]
    eigenfaces = eigenfaces.T

    # Plot one eigenface to check
    plt.imshow(eigenfaces[:, 0].reshape((h, w)), cmap='gray')
    plt.title('Eigenface 1')
    plt.show()

    return eigenfaces, num_eigenfaces


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    '''
    For all images, compute their eigenface-coefficients with respect to the given amount of eigenfaces
    :param images: 2D-matrix with a set of images as row vectors, shape (#images, #pixels)
    :param eigenfaces: 2D-array with eigenfaces as row vectors, shape(#pixels, #pixels)
                       -> only use the given number of eigenfaces
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to compute coefficients for
    :return: coefficients/features of all training images, 2D-matrix (#images, #used eigenfaces)
    '''
    centered_images = images - avg
    coefficients = np.dot(centered_images, eigenfaces[:, :num_eigenfaces])

    return coefficients



def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Reconstruct the given image by weighting the eigenfaces according to their coefficients
    :param img: input image to be reconstructed, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used to reconstruct the input image
    :param h: height of a original image
    :param w: width of a original image
    :return: the reconstructed image, 2D array (shape of a original image)
    '''
    # Reshape the input image to fit in the feature helper method
    img = img.reshape(-1, 1)

    # Compute the coefficients to weight the eigenfaces
    coefficients = get_feature_representation(img, eigenfaces, avg, num_eigenfaces)

    # Use the average image as a starting point to reconstruct the input image
    reconstructed_img = avg.reshape(1, -1) + np.dot(coefficients, eigenfaces[:, :num_eigenfaces].T)

    # Reshape the reconstructed image to match the shape of the original image
    reconstructed_img = reconstructed_img.reshape(h, w)

    return reconstructed_img


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Classify an image by reconstructing it using the given number of eigenfaces,
    and then classifying the reconstructed image using the nearest neighbor algorithm.
    :param img: input image to be classified, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used for reconstruction and classification
    :param h: height of a original image
    :param w: width of a original image
    :return: the predicted label of the input image
    '''
    reconstructed_img = reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w)
    # plt.imshow(reconstructed_img, cmap='gray')
    # plt.show()
    return reconstructed_img
