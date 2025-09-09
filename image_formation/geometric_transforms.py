import cv2 as cv
import numpy as np


def main():
    # read image
    img = cv.imread('./images/original_image.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols,_ = img.shape

    # For perspective transformation (independent corner movement):
    # Define source points (original corners)
    src_points = np.float32([
        [0, 0],           # top-left
        [cols-1, 0],      # top-right  
        [cols-1, rows-1], # bottom-right
        [0, rows-1]       # bottom-left
    ])

    left_top = np.array([cols * 0.25, rows * 0.05])
    right_top = np.array([cols * 0.74, rows * 0.39])
    
    # Angle in radians
    theta = np.deg2rad(35)

    # Vector of length = rows at 35° downwards
    dx = np.cos(theta) * rows
    dy = np.sin(theta) * rows
    side_vec = np.array([dx, dy])

    # Bottom corners
    left_bottom  = left_top  + side_vec
    right_bottom = right_top + side_vec
    dst_points = np.float32([
        left_top,
        right_top,
        left_bottom,
        right_bottom
    ])
    
    # Get perspective transformation matrix (3x3)
    M = cv.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transformation
    dst = cv.warpPerspective(img, M, (cols, rows))

    # show image
    cv.imshow('Perspective Transform', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()