# BOVW
python bag-of-visual-words-imaplementation

# Install
```
pip install git+https://github.com/stamas02/BOVW
```

# Features
you can select 3 feature types: ["SURF", "SURF_e", "SIFT"].

# Clustering
The implementeation uses sklearn.cluster.KMeans for clustering.

# Example usage
```
from bovw.bovw import BOVW

dataset = [
    "example_dataset/img1.jpg",
    "example_dataset/img2.jpg",
    "example_dataset/img3.jpg",
]

model = BOVW(n_clusters=10)
model.train(dataset)
```

# Opencv NON-FREE issue
Unfortunatelly opencv-python does not automatilaccy install the implementation of the three feature types we use. If you get an error cv2 complaining about missing xfeatures2d then you have the same issue.

Solution:
In linux terminal do the following:
```
export ENABLE_CONTRIB=1
pip install cmake
sudo apt-get install qt5-default

cd PATH/TO/CLONE/OPENCV-PYTHON/REPO
git clone https://github.com/skvark/opencv-python
cd opencv-python
```

Edit the setup.py:

line 9: cmake_args.append("-DWITH_QT=4")

Change to:

line 9: cmake_args.append("-DWITH_QT=5")

after line 127 add the following line:

"-DOPENCV_ENABLE_NONFREE:BOOL=ON"

And then:
```
python setup.py install
```
