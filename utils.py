import cv2
import numpy as np

def image_step(image):
    # Display the image in the window.
    cv2.imshow('Image', image)
    # Wait for a key press
    cv2.waitKey(0)

def find_centroid(binary_image, display_image):
    """
    """
    contour, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contour[0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print(f'Centroid of shape:\n\timage x: {cx}\n\timage y: {cy}')

    # Draw the centroid on the display image.
    cv2.circle(display_image, (cx, cy), 3, (0, 255, 0), -1)

    # Draw centroid coordinates on the display image.
    cv2.putText(display_image, f'centroid: ({cx}, {cy})', (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    return (cx, cy), display_image

def find_mean_circle(binary_image, inner_size, display, centroid):
    """ Finds the radius of the mean circle. The mean circle is the circle that contains half of the 
    pixels in the difference image between the binary image and the inner circle.
    """
    # Create an empty binary image to store the inner circle.
    inner = np.zeros(binary_image.shape, np.uint8)  
    # Draw the inner circle on the binary image.
    cv2.circle(inner, centroid, inner_size, (255, 255, 255), -1)

    # Subtract the inner circle from the binary image.
    diff = binary_image - inner

    # Find the mean circle.
    mean_found = False
    mean_size = inner_size - 1
    # The target is half the number of pixels in the difference image.
    target = np.sum(diff == 255) // 2

    while mean_found == False and mean_size < binary_image.shape[0]//2:
    
        # Increment circle size.
        mean_size += 1

        # Create an empty binary image to store the mean circle.
        mean = np.zeros(binary_image.shape, np.uint8)

        # Draw a mean circle on the binary image.
        cv2.circle(mean, centroid, mean_size, (255, 255, 255), -1)

        # Draw the mean cirle contours on a copy of the display image.
        display_copy = display.copy()
        cv2.circle(display_copy, centroid, mean_size, (255, 255, 255), 1)

        # Draw the mean circle size as text in the top left corner of the display image.
        cv2.putText(display_copy, f'mean circle radius: {mean_size}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Display the image in the window.
        cv2.imshow('Image', display_copy)
        cv2.waitKey(1)

        # Check if the mean circle is found.
        if np.sum(binary_image - mean == 255) <= target:
            print('Mean circle found:\n\tradius:', mean_size)
            mean_found = True

    return mean_size, display_copy

def find_inner_circle(binary_image, display_image, centroid):
    """
    """
    # Find the inner circle. 
    inner_found = False
    inner_size = 0

    while inner_found == False and inner_size < binary_image.shape[0]//2:

        # Increment circle size.
        inner_size += 1

        # Create an empty binary image to store the inner circle.
        inner = np.zeros(binary_image.shape, np.uint8)

        # Draw the inner circle on the binary image.
        cv2.circle(inner, centroid, inner_size, (255, 255, 255), -1)

        # Find the contours of the inner circle.
        inner_contour, _ = cv2.findContours(inner, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Create a mask of the inner circle pixels. 
        inner_mask = (inner == 255)

        # Check if inner circle is found.
        if 0 in binary_image[inner_mask]:
            print('Inner circle found:\n\tradius:', inner_size)  
            inner_found = True

        # Draw the inner circle contours on a copy of the display image.
        display_copy = display_image.copy()
        cv2.drawContours(display_copy, inner_contour, -1, (0, 255, 0), 1)

        # Draw the inner circle size as text in the top left corner of the display image. 
        cv2.putText(display_copy, f'inscribed circle radius: {inner_size}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Display the image in the window.
        cv2.imshow('Image', display_copy)

        # Close the window.
        cv2.waitKey(1)

    return inner_size, display_copy

def find_outer_circle(binary_image, display_image, centroid):
    """
    """
    # Find the outer circle.
    outer_found = False
    outer_size = binary_image.shape[0]//2

    while outer_found == False and outer_size > 0:
        
        # Decrement circle size.
        outer_size -= 1
    
        # Create an empty binary image to store the outer circle.
        outer = np.zeros(binary_image.shape, np.uint8)
    
        # Draw the outer circle on the binary image.
        cv2.circle(outer, centroid, outer_size, (255, 255, 255), -1)
    
        # Find the contours of the outer circle.
        outer_contour, _ = cv2.findContours(outer, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        # Check if the outer circle is found.
        # Create a mask of the outer circle pixels. 
        outer_mask = (outer == 0)
    
        if 255 in binary_image[outer_mask]:
            print('Outer circle found:\n\tradius:', outer_size)  
            outer_found = True
    
        # Draw the outer circle contours on a copy of the display image.
        display_copy = display_image.copy()
        cv2.drawContours(display_copy, outer_contour, -1, (0, 0, 255), 1)

        # Draw the outer circle size as text in the top left corner of the display image. 
        cv2.putText(display_copy, f'circumscribed circle radius: {outer_size}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
        # Display the image in the window.
        cv2.imshow('Image', display_copy)
    
        # Close the window.
        cv2.waitKey(1)

    return outer_size, display_copy