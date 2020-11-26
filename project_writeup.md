## Advanced Lane detection Project Writeup

--

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/output.png "Example 1"
[image2]: ./output_images/output_!.png "Example 2"
[image3]: ./output_images/edge_detection.png "Edge Detection"
[image4]: ./output_images/gradient.jpg "Gradient Result"
[image5]: ./output_images/test_undist.jpg "Undistorted Image"
[image6]: ./test_images/camera_calibration_test.png "Distorted Image"
[image7]: ./output_images/undistorted_road.png "Distorted Road Image"
[video1]: ./output_images/output_vide.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in camera_calibration_configuration() function in advanced_lane_detection.py.

The function loads a series of chessboard images, and for each image finds the corners of the chessboard. The return value of the function are two arrays 'objpoints' and 'imgpoints'. Object points contains 3d points in real world space, while image points containes 2d points on an image plane.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Dictorted Image][image6]
![Undistorted Image][image5]

### Pipeline (single images)

#### 1. Distortion-corrected of an image image.

Once we have the calibrated camera, we can calculate the required parametes for distortion correction of an image (distortion coefficients and the camera matrix). With that we can use OpenCv function undistort to apply the distortion correction
![Undistorted Road Image][image7]

#### 2. Edge Detection
For edge detection as an input I use an undistorted image from the previous step. The edge detection is done with the combination of the following methods: the combination of x, y, direction and magnitude threshold and and HSL color space thresholding. 

For the gradient computation I use the overlaping outputs of x and y sobel operators as well as direction and magnitude threshold of the image. The binary output can be shown in the following image:

![Gradient][image4]

Since the gradient thresholding method has it own limitations and I needed the algorithm to work in different light condition I used HSL thresholding. Specificaly the saturation and light component. The result of the overlapping S and L thresholds as well as their individual outputs can be seen in the image below:

![Colorspace Output][image3]


#### 3. Perspective Transformation

The function warper which is used to perform perspective trasformation can be found in advanced_lane_detection.py lines 41-48. The function takes as inputs source and destination matrix and performs the perspective transformation. The source matrix is a quadrilateral poligon that is located in the reqion where the lane lines are most likely to be in an image. The destination matrix is just a rectangular figure.

```python
    src = np.float32([(706, 450),  # top right
                      (1165, 687),  # bottom right
                      (248, 687),  # bottom left
                      (586, 450)])  # top left

    dst = np.float32([(input_image.shape[1] * 0.9, 0),
                      (input_image.shape[1] * 0.9, input_image.shape[0]),
                      (input_image.shape[1] * 0.1, input_image.shape[0]),
                      (input_image.shape[1] * 0.1, 0)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 706, 450      | 1152, 0       | 
| 1165, 687     | 1152, 720     |
| 248, 687      | 128, 720      |
| 586, 450      | 128, 0        |

The image below contains as part a warped binary image that is the output of the edge detection step.

#### 4. Lane line detection

The lane detection step takes as input a thresholded warped image, that is a bird-view binary image where the white pixels represent lane lines. The function find_lane_pixels can be found in advanced_lane_detection.py line 160. It uses a sliding window technique that iterates through the defined windows that might or might not contain lane lines. After it fit_polynomial function at line 242, takes the output of the find_lane_pixels function which are arrays of x and y values for both of the lines and the resulted output image (shown below)

![Detection Result][image2]

#### 5. Curve radius

This is done using the measure_curvature_real function in advanced_lane_detection.py file, at line 402-418. It applies the formula for calculation of the curve radius based on the fitted parameters of the left and the right lane. The results can be seen in the project output video.

#### 6. Back to real world

After applying the perspective transformation and ploting the positions of the lane lines, it was required that I apply an inverse trasformation on the warped image to project those points back on the road. The result can be seen in the image below

![Output][image1]

---

### Pipeline

#### 1. This is the output of the lane detection.

The detection uses a combination of sliding window technique and search from prior. The search from prior is used to take the load of the processing part, since it is not required to do a blank search on every frame in the video. However, since the search from prior technique is not 100% accurate, in the interest of percision I implemented a frame counter and for every 7th frame in the video I do a blank search and then search from prior

Here's a [link to my video result](./output_images/output_video.mp4)

---

### Discussion

#### 1. Problems
One of the problems during the development of the project was the parts of the road where the color changes. In the result video the lane detection is a bit wobly in there, but not so much that it completly misses the lane. This could be improved with the impalementation of a function to average the lane position from previous measurements 
