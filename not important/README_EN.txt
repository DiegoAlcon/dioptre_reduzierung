What this file is meant to: This is a plain text doc which is aimed mainly to be a brainstorm place for the project discussed on the 19.03.2024 at 13.30 pm in R&D Alcon.
 
Background: A series of images are available to us which portray scanned Waffers. Each of this images has a Dioptre factor which is defined as the inverse of the distance the lense can reach.
            Through image processing techniques, a series of parasitive circles were identified and deleted from the images. This reduced the Dioptre Factor by an unknown amount. 

Scope: During the filling of Paris lenses, air bubbles of different sizes are created. These can be removed after filling using a vacuum oven.
       However, this reduces the volume in the lens and therefore also the diopter. 
       To counteract this, the idea is to overfill the lens by the volume of the air bubble.
       Initial attempts have been made to measure the diameter of the bubbles manually using a microscope and determine the volume of the air bubble.
       However, the result was not significant. 
       Which result? The result concerning the accuracy of the results or the amount of volume to overfill?
       One question to ask is whether it is really a spherical appearance and whether the appearance of the entire lens has an influence on this.
       --> The aim of this work is to select and investigate more modern methods of image processing to find out if there is a correlation (between the diameter of the bubbles and the reduced diopter) or
           if a prediction of the dioptric change of the bubble removal process can be made. 

Main Task: Given a database of these images (before and after the parasitive circle removal) with their associated Dioptre factor (labels), construct a simple algorithm which identifies 
           any given image of this nature (before removing the bubble) and predicts the Dioptre facto reduction.

Usage: 
    Input: Image with "Blase" (aka. bubbles or circles) 
    Output: Dioptre factor reduction.

Training: 
    Input: Images with and without "Blase" with their respective labels. 

Take the following steps into account: 

Preprocessing the Images:
    Read the images.
    Apply any necessary image processing techniques (e.g., noise reduction, edge detection) to enhance features.
    If not already done, identify and remove parasitic circles from the images.
Feature Extraction:
    Extract relevant features from the preprocessed images. These features will be used for predicting Dioptre factors.
    Possible features include:
        Intensity Distribution (Histogram): Analyze pixel intensity values across the image.
        Geometric Features: Extract properties of the identified circles (e.g., area, perimeter, circularity).
        Texture Features (e.g., Haralick Features): Describe the texture patterns within the image.
        Other Domain-Specific Features: Consider any additional features relevant to your specific dataset.
Dataset Splitting:
    Divide the dataset into training and testing subsets.
Machine Learning Model Selection and Training:
    Choose an appropriate machine learning algorithm (for regression, since we are dealing with a continous output problem).
    Train the model using the training data and the extracted features.
    Tune hyperparameters if necessary.
Model Evaluation:
    Evaluate the model’s performance using the testing data.
    Use appropriate evaluation metrics (e.g., mean squared error for regression, accuracy for classification).
Predicting Dioptre Factors for New Images:
    Given a new image:
        Preprocess it (similar to the training images).
        Extract relevant features.
        Use the trained model to predict the Dioptre factor.

Questions that you should be asking:
    Which programming language to use? --> Python
    In which format will the images be delivered? --> .jpg
    Is this meant to further develop into an online application?
    If intended for online usage, what time-consuming criteria should be taken into account?
    Is this problem well suited for machine learning techniques?
    If so, which libraries should be used? Which programming level is required?
    Which type of machine learning algoritm should be used? Regression? Classification? 

Think of some hypothesis about what could affect the Dioptre factor of each image (important for feature extraction)
    The presence of this parasitic circle: [1= presence, 0= absence]
    The size of this parasitic circle: radii
    The shape of this parasitic circle: momentarily we will assume that all are circles
    The amount of these parasitic circles: 
    The average pixel intensity of this parasitic circle. 

Brainstorm a generalized cascode for a possible implementation.
    Image Pulling:
        Construct a function that reads the images from a folder.
        Before construction of main arrays, match dimensions of all images.
            For doing this average a large number of Waffer's diameters (d) to construct images of size dxd centered at the Waffer's center.
        Construct two arrays of the form: 
            with_bubble = x_dim x y_dim x n_images or x_dim x y_dim x z_dim (3) x n_dim (if not already in gray scale)
            without_bubble  = x_dim x y_dim x m_images or x_dim x y_dim x z_dim (3) x n_dim (if not already in gray scale)
    Image Preprocessing:
        Convert them to gray scale.
        Perform artifact rejection:
            Images can also be noisy. When dealing with noisy images either get rid of the noise or delete sample.
            Some rejection criteria include (for this example in specific):
                Lack of a well defined waffer centered in the image.
                Deviation from the total average energy superior to certain threshold.
            To tackle this inconveniences, apply a Butterworth High Pass Filter and evaluate betterments.
        Mix both arrays on:
            images = x_dim x y_dim x (n+m)m_images
            training_images = 0.8*(images)
            validation_images = 0.2*(images)
        Construct a label list as:
            labels = [diopter1, diopter2, ..., diopter(n+m)]
            training_labels = 0.8*(labels)
            validation_labels = 0.2*(labels)
        Consider that to implement a classfication task, the output's resolution must be large. 
    Image Training:
        Decide whih programming language to use: Python in VS connected to a git respository.
        Decide which algorithm to use: 
            Visual Geometry Group (Convolutional Neural Network) (Classification-related)
                This implementation is time-consuming, consider a faster architechture for online applications. This implementation is not well suited since we are dealing with a regression problem.
                But perhaps it is valuable to implement since a good enough quasi-continous resolution can be achieved by dividing the output's space into sufficiently small intervals, each representing a classification.
            Regression Neural Network coupled with a Hough Transform feature extraction (HTFT)
                (This implementation is well suited for regression problems)
                HTFT:
                    Series of features that can be extracted by using information provided by the Hough Transform, like:
                        - Presence of circles.
                        - Position of the circle (x, y) coorinates.
                        - Radii of the circle.
                        - Number of circles.
                        - Average pixel intensity within the circle.
                    All this features should be grouped in a characteristics vector and normalized to the range [0 1] without disturbing the original relative deviation. 
                    The inner input of the model (after applying the Hough Transform to the original image) is this feature vector.
                    Construct a series of fully connected layers and adjust the weights of each feature by applying Backpropagation with an error function based on Least Mean Squares, to obtain good regression results.
                    Check different models' implementations (linear, quadratic, etc.)
                    Obtain a mathematical model to correlate each input image with the change of the dioptre factor.
        Declare net's hyperparameters: 
        