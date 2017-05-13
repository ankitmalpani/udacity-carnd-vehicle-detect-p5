# udacity-carnd-vehicle-detect-p5
Vehicle detection using computer vision and ML Classification (Udacity CarND Project 5)



**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./sample_images/pipeline_1.png "pipeline1"
[image2]: ./sample_images/pipeline_4.png "pipeline4"
[image3]: ./sample_images/pipeline_5.png "pipeline5"
[image4]: ./sample_images/pipeline_2.png "pipeline6"
[image5]: ./sample_images/ycr_orig.png "ycr_orig"
[image6]: ./sample_images/ycr_hog0.png "ycr_h0"
[image7]: ./sample_images/ycr_hog1.png "ycr_h1"
[image8]: ./sample_images/ycr_hog2.png "ycr_h2"
[video1]: ./output.mp4 "video"

### A Retrospective
I should acknowledge that there was plenty of pre-coded work from the chapter exercises for this project. So most of the code is quite similar to what is seen in the class project. For this reason, the ready setup made this project look quite easy at the beginning. However, as I started working on the intricacies and reading and understanding the various concepts involved in the project, it turned out to be quite challenging.

### Code Structure and Notes
The code in my submission (vehicle_detect.ipynb) has been structured in similar way like my last few projects so that it is easier to follow. This should serve as an index for the reader and should serve as guide on which cells include what code.
1. Imports -  This is where I import all the necessary libs needed through out the project
2. Functions - This cell has all the methods used through out the project (except pipeline). This can be thought of as an extended and modified version of `lesson_functions.py` from the chapters
3. Data load - includes code to read in all file paths using glob
4. Data exploration - I have kept this unexecuted in the final version for cleanliness
5. Feature Extraction - This cell includes all code for feature extraction, feature scaling and train-test split
6. Feature exploration and shuffle
7. Classifier definition and accuracy check - This is where I create the classifier, train it and check its accuracy
8. Pipeline - This is the final pipeline definition
9. Some test cells
10. Video generation
* Note: I have also changed the format of readme a little bit for some additional info. For example, I have added a section classifier, which I think was missing here.

###Histogram of Oriented Gradients (HOG)
This was a very interesting concept and it took me going over the chapter a couple of times to understand this well. I also referred to the additional video resource to get good clarification on this. I think this get even more interesting with numbers when `hog sub-sampling` is introduced in the lecture.
The code for this step is contained in the second code cell (with rest of the methods and functions) of the IPython notebook  `get_hog_features.py`.  This is pretty similar to what we wrote in class exercise.
Here is an example of one original image, converted `YCrCb` image and hog transforms for each YCrCb channels with `orient=9` , `pixels_per_cell=8` and `cells_per_block=2`

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

#### Final choice of HOG parameters.
After exploring different combinations of params for colorspace `HSV` and `YCrCb` - I decided to keep these the same as lectures and it proved to be quite accurate for the classifier. the main idea for me was to get a general sense of object detection via its output.
I really liked the outputs for `HSV` colorspace initially, and I chose that as the colorspace to go with, however after running tests with classification (described later) - I started using `YCrCb` where the classifier performed better (though) the object detection was similar to HSV space.

### Classifier
I took the suggestion from the chapter videos and decided to go for an SVM. I started off with running a GridSearch on multiple parameters including the Kernel. I started testing with `Kernel:{'rbf', 'linear'}` and C param `C:{0.5, 1, 10}`. After some iterations and trials on the classifer i decided to proceed with a `LinearSVC` and `C=0.5` in my final Classifier. My accuracy stayed quite consistent around ~99%. With my initial trials with `rbf` kernel, I wasn't able to get good results with my test images though.

#### use of HOG features
I trained a Linear SVM as seen in Cell 7, however to see the use of HOG features, one can see it in the Cell 5 with `Extract Features` which calls a method in Cell 2 called `extract_features()`. One can see that the value of param `hog_feat` is set to `True` which basically triggers the code to extract HOG features and use it in the training set.

### Sliding Window Search
This code was pretty much as seen in the class exercise. One can see this in the method `sliding_window` in the cell with all the methods (Cell 2). The calculations of window sizes took a few iterations and trials where I tried to estimate the right window sizes for horizon views ( Cars closer to y values of 350 - 450 ) and then slowly increasing the size of the windows.
I should mention that i ended up not using this method in favor of the `find_cars()` method from the chapter, which basically helps efficiently extract hog features and then subsample the windows using the scale. However, this was a similar problem, just in terms of scales. I ran a few different iterations/trials for these before I got good results on all test images. This can be seen in the cell with the `pipeline()` definition and here is the final set for [ystart, ystop and scale] i used
`iterations = [(400, 470, 1), (400, 600, 1.5), (400, None, 2.0), (400, None, 2.5)]`

Here are some images that show the results for these values

![alt text][image3]
![alt text][image2]

#### Working Pipeline

Ultimately I searched on 4 scales (as shown above) using `YCrCb` 3-channel HOG features, spatially binned color and histograms of colors in the feature vector. This can be seen in `extract_features()` method in Cell 2. I also should mention the use of an averaging heatmap for last 15 frames. The final pipeline provided a good result on test images and the final video.  Here are some example images:

![alt text][image1]
![alt text][image4]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and used it towards averaging heatmap numbers for last 15 frames, and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I mentioned earlier in the retrospective. The project was quite challenging once I started understanding the varioius steps needed in this pipeline. I had to refer to some external resources to understand some concepts like Histogram of Oriented Gradients. I also has to re-read the code for hog sub sampling a couple of times to get the idea. Moreover, I think this project is far from done. I noticed one of the instructors mentioning the use of deep learning for such tasks. Also, I see some students/mentors have taken such approaches in the past. I think my next challenge would be to explore solving this with Deep Learning.
I also want to mention and acknowledge some external resources I used while working on this project along with a Thanks to the mentors on forums who always help a lot:
* https://www.youtube.com/watch?v=7S5qXET179I
* http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
* https://discussions.udacity.com/t/svm-overfitting/234904
* https://discussions.udacity.com/t/accuracy-of-svm-is-99-on-test-test-but-not-detecting-a-single-car-in-test-images/237614
* Next steps: https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6
