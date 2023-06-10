from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)
    return kernel


def slow_convolve(image, kernel):
    # Image dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Kernel dimensions
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Calculate padding dimensions
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Prepare an empty array to store the result
    convolved = np.zeros_like(image)

    # Flip the kernel both horizontally and vertically
    kernel = np.flip(kernel)

    # Check if image is grayscale or color
    if len(image.shape) == 2:  # Grayscale
        # Create a padded copy of the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Convolution operation
        for i in range(image_height):
            for j in range(image_width):
                # Element-wise multiplication of the kernel and the image pixels
                convolved[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])
    else:  # Color
        # Create a padded copy of the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

        # Convolution operation
        for i in range(image_height):
            for j in range(image_width):
                for k in range(image.shape[2]):  # Iterate over each color channel
                    # Element-wise multiplication of the kernel and the image pixels
                    convolved[i, j, k] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width, k])

    return convolved


if __name__ == '__main__':
    k = make_kernel(10, 5)

    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # Blur the image
    blurred = slow_convolve(im, k)

    # Subtract the blurred image from the input image
    subtracted = im - blurred

    # Add the result to the input image
    added = im + subtracted

    # Clip the values to the range [0, 255]
    clipped = np.clip(added, 0, 255)

    # Convert the array to np.uint8
    clipped = clipped.astype(np.uint8)

    # Save the result
    result = Image.fromarray(clipped)
    result.save('output.jpg')
