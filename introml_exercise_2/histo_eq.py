# Implement the histogram equalization in this file

from PIL import Image
import numpy as np

image = Image.open("hello.png")
image_array = np.array(image)

histogram = np.zeros(256, dtype=int)
rows, cols = image_array.shape

for i in range(rows):
    for j in range(cols):
        pixel_value = image_array[i, j]
        histogram[pixel_value] += 1

cdf = np.cumsum(histogram)
cdf_normalized = cdf / np.sum(histogram)

Cmin = np.min(cdf_normalized)
transformed_image = ((cdf_normalized[image_array] - Cmin) / (1 - Cmin) * 255).astype(np.uint8)

result_image = Image.fromarray(transformed_image)
result_image.save("kitty.png")


