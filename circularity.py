import time
import cv2
import numpy as np

from utils import image_step, find_centroid, find_mean_circle, find_inner_circle, find_outer_circle

# Load an image from file
image = cv2.imread("ruby_beach.jpeg", cv2.IMREAD_GRAYSCALE)

# Create a window to display the image
cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)

image_step(image)

# Binary threshold. 
_, thresh = cv2.threshold(image,90,255,cv2.THRESH_BINARY_INV)

image_step(thresh)

# Close filter.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

image_step(closed)

# Open filter. 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

image_step(opened)

# Find a bounding box of the object. 
contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the image to the bounding box.
padding = 0.1
cropped = opened[y - int(padding*h):y + h + int(padding*h), x - int(padding*w):x + w + int(padding*w)]
image = image[y - int(padding*h):y + h + int(padding*h), x - int(padding*w):x + w + int(padding*w)]

image_step(cropped)

# Create a display image.
display = image.copy()
display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

centroid, display = find_centroid(cropped, display)
inner_radius, display = find_inner_circle(cropped, display, centroid)
outer_radius, display = find_outer_circle(cropped, display, centroid)

# Computer radius error. 
radius_error = inner_radius / outer_radius
print(f'ISO roundness:\n\t{radius_error:.4f}')

# Draw error as text in the top left corner of the display image.
cv2.putText(display, f'ISO roundness: {radius_error:.4f}', (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)

image_step(display)

cv2.destroyAllWindows()

