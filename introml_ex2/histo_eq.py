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

# Compute the cumulative distribution function
cdf = np.zeros(256, dtype=float)
cdf[0] = histogram[0]

for i in range(1, 256):
    cdf[i] = cdf[i - 1] + histogram[i]

# Normalize the cumulative distribution function
cdf_normalized = cdf / np.sum(histogram) #that values are between 0 and 1

Cmin = np.min(cdf_normalized)
transformed_image = ((cdf_normalized[image_array] - Cmin) / (1 - Cmin) * 255).astype(np.uint8)

result_image = Image.fromarray(transformed_image)
result_image.save("kitty.png")

#background has low variation in pixel intensities, so these values can be redestributed, to cover a wide range of intensities. this leads to stronger intensitie boundaries


# Check histogram sum
histogram_sum = np.sum(histogram[:90])
print("Histogram sum:", histogram_sum)

# Check cumulative distribution sum
cdf_sum = np.sum(cdf_normalized[:90])
print("Cumulative distribution sum:", cdf_sum)

# Compare cumulative distribution sum with expected value
expected_cdf_sum = 0.001974977
if np.isclose(cdf_sum, expected_cdf_sum):
    print("ja correct.")
else:
    print("das passt nicht")

