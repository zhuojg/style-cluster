# Design Style Cluster with ResNet34 and K-means
This project tries to extract image features with CNN and cluster them with K-means or other clustering algorithms.  
The result seems to be reasonable with data from Tongji Tezign Design A.I Lab (https://sheji.ai).

## Module
### style_model.py
CNN model for feature extracting.  
ResNet34 is used for this purpose. According to Gtys L A[1], the style feature is considered as the Gram Matrix of the feature maps from last convolutional layer. 

### feature.py
Extract features using the model from style_model.py and reduce the dimension using PCA.  
Run this file to get the features for clustering.

### demo.py
Cluster the features extracted from job.py and visualize the results.

## How to use
Run feature.py firstly to get features and run demo.py to get the results.  
**Remember to change the save_path and result_path in each file.**

## Result
You can get the result in result directory. One image for one category.

## Reference
[1] Gatys L A , Ecker A S , Bethge M . A Neural Algorithm of Artistic Style[J]. Computer Science, 2015.