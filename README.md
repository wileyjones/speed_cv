# Find speed of a moving vehicle from forward-facing video

![Derivative Image from real_time.py](https://github.com/wileyjones/speed_challenge/blob/master/derivative_img.jpg)

# Approach
Using basic signal processing and computer vision, find an image derivative and map it to a vehicle speed.

There are two files included, `real_time.py` and `model.py`. Real-time is accurately named as it is runs a frame-by-frame. It can even show the derivate images (check line 165 in the code to play with this). The other is `model.py` which processes the data as fast as possible ~5x speed of `real_time.py`. This file could be extended to include additional ML/statistics post-processing and model generation.

# Outline
1. Read in current frame, store in a queue
2. Read in next frame, append to queue, and call `derivative` function
  * `derivative` works by first taking (frame(n+1) - frame(n)) with grayscale images
  * These width x height 1-channel pixel matrices show difference between two images
  * To normalize data, sqrt(image^2) magnitude is taken
  * Next to get the matrix to score-like scalar, the sum of rows and columns is taken
3. Pass these image energy deltas as (I like to refer to them) through filters
  * I noticed that the spike1y responses were around areas of velocity, but needed flattening
  * The Savitzky-Golay is particularly ideal for this application, as a smoothing filter that has great high-frequency rejection without damaging general signal shape
4. Plot data and manually tweak number of filter passes, windowing, polynomial fit
  * Optimized for generality as opposed to RMSE, avoiding overfitting

# Comments
The general shape achieved by this approach is quite compelling for a few main reasons:

1. General shape is shown by simple (low cost) matrix computations and filtering
2. Accuracy and generality could be extended with ML techniques and more data
3. These methods requires no training and can be implemented in real-time as in the `real_time.py` file

It is quite compelling that a scalar value can be computed from images alone and with 0 additional contextual data it provided a general understanding of that velocity profile of a vehicle was changing. This approach could be furthered with machine learning techniques that could make sense of the complex mapping between our image energy delta scalars and the speed, also a scalar. Providing additional contextual data could serve as a way of improving the generality of this approach.

# Further Testing and Validation
To test the models and files, simple link the correct JSON file and video .mp4 into the file and it will run!
If you'd like to play around with it some more, feel free to go into the Jupyter notebook, it's the best way of
working on changing the model without having to recompute the video every time.

* Python 3 with: Numpy, OpenCV, matplotlib, imageio, json, and other more standard libs
