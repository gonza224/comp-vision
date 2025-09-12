
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_image_for_plotting(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def main():
    # Read image with opencv
    original_img = get_image_for_plotting('./images/original_image.jpg')
    transformed_img = get_image_for_plotting('./images/transformed_image.jpg')

    rows,cols,_ = original_img.shape

    # We are going to use warp perspective to transform the image.
    # Basically map the 4 corners of the original image to the 4 corners of the new image.

    # Source points. These are the corners of the original image. (Basically a 4x2 matrix)
    src_points = np.float32([
        [0, 0],           # top-left
        [cols-1, 0],      # top-right  
        [cols-1, rows-1], # bottom-right
        [0, rows-1]       # bottom-left
    ])

    # Destination corners moved by percentage (by eye). These are the corners of the new image.
    left_top = np.array([cols * 0.25, rows * 0.05])
    right_top = np.array([cols * 0.75, rows * 0.39])
    left_bottom = np.array([cols*0.65, rows*1.05])
    right_bottom = np.array([cols*1.12,rows*1.35])

    # Destination points. (Basically a 4x2 matrix)
    dst_points = np.float32([
        left_top,
        right_top,
        right_bottom,
        left_bottom,
    ])
    
    # Get perspective transformation matrix (3x3)
    M = cv.getPerspectiveTransform(src_points, dst_points)
    print(f"Perspective transformation matrix: {M}")
    
    # Apply perspective transformation
    reverse_eng_img = cv.warpPerspective(original_img, M, (cols, rows))

    # Show results side by side
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(original_img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(transformed_img)
    axs[1].set_title("Transformed")
    axs[1].axis('off')

    axs[2].imshow(reverse_eng_img)
    axs[2].set_title("Reverse Engineered")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()