import function
import numpy as np
import pandas as pd
from numpy import ndarray, array
from sklearn.mixture._base import BaseMixture
from PCA_reduction.PCA import PcaFeatsWrapper
from base_factory import BaseFactory
from utility.random_index import randIndexComputing


class BaseModelFactory(BaseFactory):

    def __init__(
            self, models: list, candidates: list[float],
            fitNdTesting: function, X: pd.DataFrame, y: ndarray
    ):
        self._pcaHandler: PcaFeatsWrapper = PcaFeatsWrapper()
        self._models: list = models
        self._candidates: list[float] = candidates
        self._fitNdTesting: function = fitNdTesting
        self._X: pd.DataFrame = X
        self._y: ndarray = y
        self._outcomeCollection: ndarray[dict[str, float]] = np.array([])

    def tuningProcess(self, returnBestModel: bool = False) -> tuple:

        # at the end of the cycle below this array shall be populated with the models R.I
        randomIndexes: ndarray[float] = array([])

        for idx in range(0, len(self._models)):
            # train the model and see how the it's doing with the training data
            prediction = self._fitNdTesting(self._models[idx], self._X)

            # compare the model prediction with the actual prediction and store it in a collection
            randomIndexes = np.append(randomIndexes, randIndexComputing(self._y, prediction))

        # find the index related to the max values
        bestIdx = np.argmax(randomIndexes)

        # find the right hyper-parameter
        hyperParam: float = self._candidates[bestIdx]

        # find the best model
        bestModel: BaseMixture = self._models[bestIdx]

        # return a tuple containing the best hyper-parameter,
        # all the random indexes and if required the best model
        if not returnBestModel:
            return hyperParam, randomIndexes
        else:
            return hyperParam, randomIndexes, bestModel

    def modelsBuilder(self):
        # retrieve the data after the dimensionality changes (PCA)
        dataFrameCollection: list[pd.DataFrame] = self._pcaProcess()

        for idx in range(0, len(dataFrameCollection)):
            # for each dimensionality reduction a tuning process is instantiated
            tempTup: tuple[float, ndarray[float]] = self.tuningProcess()

            # append the result of the tuning process
            self._outcomeCollection = np.append(self._outcomeCollection, {
                "dimensions": idx + 2,
                "hyper-param": tempTup[0],
                "rand-index": np.max(tempTup[1])
            })

        return self._outcomeCollection

    def _pcaProcess(self) -> list[pd.DataFrame]:
        return self._pcaHandler.pcaIterable(self._X)

    def updateDataSet(self, X_new: pd.DataFrame, y_new: ndarray):
        self._X = X_new
        self._y = y_new
