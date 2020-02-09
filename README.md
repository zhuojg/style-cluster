# Design Style Cluster with ResNet34 and K-means

This project tries to extract image features with CNN and cluster them with K-means.  
The result seems to be reasonable with data from Tongji Tezign Design A.I Lab (<https://sheji.ai).>

## Setup
* Create python virtual environment
```shell script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Demo
* Download preprocessed features `pca_features.pkl` from  
https://drive.google.com/open?id=1yeMFaNly7pZRGcTWPEmAvHZ2pY4gSRJ7   
and move it to `./result`
* Download data from  
https://drive.google.com/drive/folders/1cdYHtQyOMfoZCpaXiiMcRK-mzH4NqLEG?usp=sharing  
* Unzip data to `./` and rename the folder to `data`  
* Run demo as follow, remember to change args according to your environment.  
```shell script
python demo.py --data_path './data' --feature_path './result/pca_features.pkl' --save_path './result'
```

| args | description |  
| :------  | :------ |  
| data_path  | path to data |
| feature_path | path to feature file |
| save_path | path to save result |  

* This figure shows images from 1 of 15 clusters.  
![result](https://raw.githubusercontent.com/zhuojg/style_cluster/master/result/0.png)


## Create your own features
* You can also create your own image style features using feature.py.
* Run feature.py as follow, remember to change args according to your environment.
```shell script
python feature.py --result_path './result' --data_path './data'
```

| args | description |  
| :------  | :------ |  
| data_path  | path to data |
| result_path | path to save result |

* After the feature is created, you can run demo.py to check the result.  

## Reference

[1] Gatys L A , Ecker A S , Bethge M . A Neural Algorithm of Artistic Style[J]. Computer Science, 2015.
