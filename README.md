# Design Style Cluster with ResNet34 and K-means
This work tries to extract image features with CNN and cluster them with K-means or other clustering algorithms.  
The result seems to be reasonable with data from Tongji Tezign Design A.I Lab.

## model - style_model.py
CNN model for feature extracting.

## job.py
Extract features using the model from style_model.py and reduce the dimension using PCA.  
Remember to change the image_path and result_path!  
Run this file to get the features for clustering.

## demo.py
Cluster the features extracted from job.py and visualize the results.  
Remember to change the image_path and result_path!

## How to use
Run job.py firstly to get features and run demo.py to get the results.

## Result
You can get the result in result directory. One image for one category.