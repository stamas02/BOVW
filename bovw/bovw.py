import logging
import math
import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import cv2


class BOVW:
    """
    Bag of visual words implementation.

    Attributes
    ----------
        feature_extraction_method: str,
            Supported methods: "SURF", "SURF_e", "SIFT". 
            Note that SURF_e refers to the extended 128 dim SURF

        n_clusters: int,
            The number of cluster centres used in clustering. 
            This will be the dimensionality of the final descriptor.
            if -1 then the value is sqrt(num_features) where number of 
            features is the number of SWIFT or SURF
            extracted from the set of training images.

        feature_per_image: int Oprional,
            Number of features per image. If -1 then all the features will be used. 
            If an image does not have this number of features then all features 
            will be used for that image. In other words the result is not guaranteed 
            to be length of img_number*feature_per_image but it is
            guaranteed that img_number*feature_per_image is the upper bound.

        max_kmeans_iteration: int Optional,
            Number of kmeans iteration to be performed during training

        kmeans_n_init: int Optional,
            Number of time the k-means algorithm will be run with
            different centroid seeds. The final results will be the best
            output of n_init consecutive runs in terms of inertia.

        max_feature_cnt: int Optional,
            maximum number of features to be used for k-means training. If
            less then the available features then the subset will be uniformly
            randomly selected. 
    """

    def __init__(
        self,
        feature_extraction_method="SURF",
        n_clusters=-1,
        feature_per_image=-1,
        max_kmeans_iteration=100,
        kmeans_n_init=10,
        max_feature_cnt=-1,
    ):
        self.max_feature_cnt = max_feature_cnt
        self.model = None
        self.kmeans_n_init = kmeans_n_init
        self.n_clusters = n_clusters
        self.max_kmeans_iteration = max_kmeans_iteration
        self.method = feature_extraction_method
        self.dim = 64 if self.method == "SURF" else 128
        self.model = None
        self.feature_per_image = feature_per_image
        pass

    def _get_extractor(self, method):
        """
        creates the feature extractor

        Parameters
        ----------
        method: str,
            Supported methods: "SURF", "SURF_e", "SIFT". 
            Note that SURF_e refers to the extended 128 dim SURF

        Returns
        -------
            The cv2 object for feature extraction

        """
        if method == "SIFT":
            return cv2.xfeatures2d.SIFT_create()
        elif method == "SURF":
            return cv2.xfeatures2d.SURF_create()
        elif method == "SURF_e":
            tmp = cv2.xfeatures2d.SURF_create()
            tmp.setExtended(True)
            return tmp
        else:
            raise ValueError("Method is not supported {0}".format(method))

    def train(self, train_image_paths):
        """
        Trains the BOVW vocabulary on the training images.

        Parameters
        ----------
        train_image_paths: List of str,
            List of paths to individual training images.

        verbosity:
            if 0 no update messages and no progressbar will be displayed
        """

        txt_template = "Extract {0} features from {1} images"
        logging.info(txt_template.format(self.method, len(train_image_paths)))
        # EXTRACT ALL FEATURES FROM TRAINING IMAGES

        features = []
        feature_extractor = self._get_extractor(method=self.method)
        desc_text = "Extracting {0} features".format(self.method)
        for p in tqdm(train_image_paths, desc=desc_text, unit="image"):
            try:
                img = cv2.imread(p, 0)
                kp, des = feature_extractor.detectAndCompute(img, None)
                if self.feature_per_image > -1 and len(des) > self.feature_per_image:
                    des = np.random.choice(des, self.feature_per_image, replace=False)
                features += list(des)
            except Exception as e:
                warning_text = "Failed to extract features for file {0} due to: {1}"
                logging.warning(warning_text.format(p, e))

        # TRAIN FAST KNN ON THE EXTRACTED FEATURES.
        if self.n_clusters == -1:
            self.n_clusters = int(math.sqrt(len(features)))

        if self.max_feature_cnt > -1 and self.max_feature_cnt < len(features):
            features = np.random.choice(features, self.max_feature_cnt, replace=False)

        logging.info("Training K-means with {} features".format(len(features)))
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_kmeans_iteration,
            n_init=self.kmeans_n_init,
            verbose=1,
        )
        self.model.fit_predict(features)
        logging.info("Train Complete!")

    def get_descriptor(self, image):
        """
        Extracts features using the trained model from an image.

        Parameters
        ----------
        image: numpy array
            the image used for feature extraction. MUST be gray scale or RGB format

        Returns
        -------
            The extracted feature.

        """
        feature_extractor = self._get_extractor(method=self.method)
        image = np.array(image)
        if len(image.shape) != 3 and len(image.shape) != 2:
            raise ValueError(
                "Image must be grayscale or RGB got {0}".format(image.shape)
            )

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, features = feature_extractor.detectAndCompute(image, None)

        descriptors = self.model.predict(features)
        hist, bin_edges = np.histogram(descriptors, bins=np.arange(self.n_clusters))
        return hist
