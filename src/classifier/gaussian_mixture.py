import pandas as pd
from numpy import ndarray, array
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture

from base_classes.base_class import BaseModelFactory
from utility.fetchedData import X_train, y_train
from utility.random_index import randIndexComputing


# tuning process of k gaussian mixtures
def tuning(X: pd.DataFrame, y: ndarray, returnBestModel: bool = False) -> tuple:
    # initialize the possible candidates (number of clusters)
    nClusters: list[int] = [2, 5, 9, 20, 35, 55, 75, 100, 150]

    # generate k gaussian mixtures each of which with a different number of clusters
    gaussianMixtures: list[BaseMixture] = [
        GaussianMixture(
            n_components=n_classes, covariance_type="diag", random_state=1629
        ) for n_classes in nClusters
    ]

    randomIndexes: ndarray = array([])

    for idx in range(0, len(gaussianMixtures)):
        # train the model
        gaussianMixtures[idx].fit(X)

        # see how the model doing with the training data
        prediction = gaussianMixtures[idx].predict(X)

        # compare the model prediction with the actual prediction and store it in a collection
        randomIndexes = np.append(randomIndexes, randIndexComputing(y, prediction))

    # find the index related to the max values
    bestIdx = np.argmax(randomIndexes)

    # find the right number of clusters
    bestNClusters: int = nClusters[bestIdx]

    # find the best model
    bestModel: BaseMixture = gaussianMixtures[bestIdx]

    # return a tuple containing the best number of clusters,
    # all the random indexes and if required the best model
    if not returnBestModel:
        return bestNClusters, randomIndexes
    else:
        return bestNClusters, randomIndexes, bestModel


# start the tuning process
tup = tuning(X_train, y_train, returnBestModel=True)

# check the random index related to the best model
print(tup)

"""
    Factory implementation
"""


class GaussianMixturesFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (number of clusters)
        self.nClusters: list[int] = [2, 5, 9, 20, 35, 55, 75, 100, 150]

        # generate k gaussian mixtures each of which with a different number of clusters
        self.gaussianMixtures: list[BaseMixture] = [
            GaussianMixture(
                n_components=n_classes, covariance_type="diag", random_state=1629
            ) for n_classes in self.nClusters
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(gaussianMixture: BaseMixture, X_t: pd.DataFrame):
            # train the model
            gaussianMixture.fit(X_t)

            # return how the model doing with the training data
            return gaussianMixture.predict(X_t)

        # set up the base class
        super().__init__(
            self.gaussianMixtures, self.nClusters, _fitNdTesting, self._X, self._y
        )

    def tuningProcess(self, returnBestModel: bool = False) -> tuple:
        return super().tuningProcess(returnBestModel)

    def modelsBuilder(self):
        return super().modelsBuilder()

    def updateDataSet(self, X_new: pd.DataFrame, y_new: ndarray):
        self._X = X_new
        self._y = y_new
        super().updateDataSet(X_new, y_new)
