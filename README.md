# GatorNum

This project was used as the final project for UF's Intro to Machine Learning Special Topics Course. Done by: Gabriel Castejon and Larry Mason, Fall 2023.

## Overview

Using ReactJS as a front-end and Python as both the back-end and ML language of choice, we developed a simple version of an optical character reader using machine learning.

The front-end takes in images in PNG or PDF format and sends it to the python back-end for pre-processing, text localization, and then a prediction using a pre-trained model of the MNIST dataset.

## Project Setup

- Step 1: Clone the repository or download it
- Step 2: Either create a virtual environment or locally install the required libraries for python:
pip install -r requirements.txt
-Step 3: Build the project using the given 'build.sh' script.

### Video Showcase:

https://www.youtube.com/watch?v=FIbR2ONn1w0

### References and Notes for the Project

Source from book:
https://github.com/ageron/handson-ml2 

Colab of a Neural Network using the Fashion dataset: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb

EMNIST Download: https://www.nist.gov/itl/products-and-services/emnist-dataset

Link to our google colab: https://colab.research.google.com/drive/1QB91eK-kfsZPERRo_GkkZ0Oq11kOSPMe?usp=sharing

Kaggle API: https://www.kaggle.com/docs/api

![image](https://github.com/slinky55/GatorNum/assets/92041237/5fe910d7-9ec9-4b61-acac-54c94a713c1c)

Preprocessing:

    Image Acquisition: Obtain the image or scanned document containing the text.

    Image Enhancement: Enhance the image quality by removing noise, adjusting brightness, increasing contrast, and applying filters to improve the clarity of the text. Techniques like Gaussian blurring, thresholding, and morphological operations can be used.

    Binarization: Convert the image to binary (black and white) to separate the text from the background. Thresholding methods like Otsu's method or adaptive thresholding can be employed to accomplish this.

    Deskewing and Rotation Correction: Correct any skew or rotation in the text to ensure it is aligned horizontally or vertically. Techniques such as Hough Transform or projection profile analysis can be used to detect and correct skew.

Text Localization:

    Region Proposal: Employ techniques like connected component analysis, contour detection, or sliding window approaches to identify potential regions containing text in the preprocessed image.

    Bounding Box Generation: Create bounding boxes or rectangles around the detected text regions to isolate them from the rest of the image.

Post-processing and Correction:

    Error Correction: Implement algorithms to correct recognition errors using context analysis, dictionary lookup, language models, or spell-checking methods.

    Text Segmentation and Formatting: Arrange recognized characters into words, sentences, or paragraphs to reconstruct the text in a readable format.
