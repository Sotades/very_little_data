import numpy as np
import cv2

def render_lane(image, corners, ploty, fitx, ):
    _, src, dst = perspective_transform(image, corners)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts = np.vstack((fitx, ploty)).astype(np.int32).T

    # Draw the lane onto the warped blank image
    # plt.plot(left_fitx, ploty, color='yellow')
    cv2.polylines(color_warp, [pts], False, (0, 255, 0), 10)
    # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result