import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC

# image size
N = 64

# Define the classifier in clf - Try a Support Vector Machine with C = 0.025 and a linear kernel
# DON'T change this!
clf = SVC(kernel="linear", C=0.025)


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
    filenames = np.sort(path)
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
    # Calculate the mean face
    avg = calculate_average_face(train)

    # calculate the maximum number of eigenfaces
    max_num_eigenfaces = num_images - 1

    # Calculate eigenfaces
    eigenfaces = calculate_eigenfaces(train, avg, max_num_eigenfaces, h, w)

    # calculate the coefficients/features for all images
    coeffs = get_feature_representation(train, eigenfaces, avg, max_num_eigenfaces)

    # Train the classifier using the features and corresponding labels
    clf.fit(coeffs, labels)

    return eigenfaces, max_num_eigenfaces, avg


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''
    return np.mean(train, axis=0)


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

    # Subtract the mean face from each image to get a centered train
    centered_train = train - avg

    # Compute the eigenfaces using SVD
    # You might have to swap the axes so that the images are represented as column vectors
    u, s, v = np.linalg.svd(centered_train.T)

    # Represent eigenfaces as row vectors in a 2D matrix and crop to the requested amount of eigenfaces
    eigenfaces = u.T[:num_eigenfaces]

    # plot one eigenface to check whether you're using the right axis
    # comment out when submitting your exercise via studOn
    # img_plot = np.reshape(eigenfaces[0], (h, w))
    # plt.imshow(img_plot, cmap="gray"), plt.colorbar()
    # plt.show()

    return eigenfaces


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

    # compute the coefficients for all images and save them in a 2D-matrix
    coeffs = np.zeros((images.shape[0], num_eigenfaces))

    # 1. iterate through all images (one image per row)
    for i in range(images.shape[0]):
        # 1.1 compute the zero mean image by subtracting the average face
        zero_mean_image = images[i] - avg
        # 1.2 compute the image's coefficients for the expected number of eigenfaces
        for j in range(num_eigenfaces):
            coeffs[i, j] = np.dot(eigenfaces[j], zero_mean_image)
    return coeffs

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
    # reshape the input image to fit in the feature helper method
    img = img.reshape(1, -1)
    # compute the coefficients to weight the eigenfaces
    coefficients = get_feature_representation(img, eigenfaces, avg, num_eigenfaces)
    # use the average image as starting point to reconstruct the input image
    recon_img = np.copy(avg)
    # reconstruct the input image using the coefficients
    for i in range(num_eigenfaces):
        recon_img += coefficients[0, i] * eigenfaces[i]
    # reshape the reconstructed image back to its original shape
    reconstructed_img = recon_img.reshape(h, w)
    return reconstructed_img


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Classify the given input image using the trained classifier
    :param img: input image to be classified, 1D-array
    :param eigenfaces: all given eigenfaces, 2D array with the eigenfaces as row vectors
    :param avg: the average image, 1D array
    :param num_eigenfaces: number of eigenfaces used to extract the features
    :param h: height of a original image
    :param w: width of a original image
    :return: the predicted labels using the classifier, 1D-array (as returned by the classifier)
    '''
    # Reshape the input image to match the size of the eigenfaces
    img = img.reshape(1, -1)
    # extract the features/coefficients for the eigenfaces of this image
    coeffs = get_feature_representation(img, eigenfaces, avg, num_eigenfaces)
    # predict the label of the given image by feeding its coefficients to the classifier
    predicted_labels = clf.predict(coeffs)
    return predicted_labels
