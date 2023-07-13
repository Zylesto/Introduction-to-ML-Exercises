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
    Processes the training data, including eigenface computation and training of the classifier
    :param labels: labels of the training images
    :param train: 2D-matrix with training images as row vectors, shape (#images, #pixels)
    :param num_images: number of training images
    :param h: height of the images
    :param w: width of the images
    :return: eigenfaces, number of eigenfaces, average face
    '''
    # Compute eigenfaces
    u, num_eigenfaces, avg = calculate_eigenfaces(train, num_images, h, w)

    # Extract features (coefficients) from training images
    features = get_feature_representation(train, u, avg, num_eigenfaces)

    # Train the classifier
    clf = SVC(kernel='linear')
    clf.fit(features, labels)

    return u, num_eigenfaces, avg


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''
    return np.mean(train, axis=1)


def calculate_eigenfaces(images, num_images, h, w):
    '''
    Calculates the eigenfaces from the given images
    :param images: 2D-matrix with images as row vectors, shape (#images, #pixels)
    :param num_images: number of images
    :param h: height of the images
    :param w: width of the images
    :return: eigenfaces, number of eigenfaces, average face
    '''
    # Compute the average face
    avg = np.mean(images, axis=0)

    # Center the images by subtracting the average face
    centered_images = images - avg

    # Compute the covariance matrix
    cov_matrix = np.dot(centered_images.T, centered_images)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize the eigenvectors
    eigenfaces = eigenvectors.T

    return eigenfaces, num_images, avg

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
    coefficients = np.dot(centered_images, eigenfaces[:num_eigenfaces, :].T)

    return coefficients



def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Reconstruct the given image by weighting the eigenfaces according to their coefficients
    :param img: input image to be reconstructed, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used to reconstruct the input image
    :param h: height of the original image
    :param w: width of the original image
    :return: the reconstructed image, 2D array (shape of the original image)
    '''
    # Reshape the input image to fit in the feature helper method
    img = img.reshape(h, w)

    # Center the input image by subtracting the average face
    centered_img = img - avg.reshape(h, w)

    # Compute the coefficients to weight the eigenfaces
    coefficients = get_feature_representation(centered_img.reshape(1, -1), eigenfaces, avg, num_eigenfaces)

    # Reshape the coefficients array to match the shape of eigenfaces
    coefficients = coefficients.reshape(-1, num_eigenfaces)

    # Use the average image as a starting point to reconstruct the input image
    reconstructed_img = avg.reshape(h, w) + np.dot(coefficients, eigenfaces[:, :num_eigenfaces].T)

    # Convert the reconstructed image to float dtype
    reconstructed_img = reconstructed_img.astype(np.float32)

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
