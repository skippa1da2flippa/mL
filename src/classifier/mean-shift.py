import pandas as pd
from numpy import ndarray
from sklearn.cluster import MeanShift

from base_classes import BaseModelFactory

"""

    # Normal mean shift 
    bandwidth = estimate_bandwidth(X_train)

    print(bandwidth)

    meanshift = MeanShift(bandwidth=bandwidth)
    meanshift.fit(X_train)
"""

"""

    # EXTRINSIC EVALUATION
    
    def tuning(X_train, y_train) -> tuple:
        randomIndexes: ndarray[float] = np.array([])
        kernelWidths: list[float] = [1.25, 9.5, 30.75, 50.25, 70.5, 100.75, 200.25, 350.5]
    
        for width in kernelWidths:
            tempMeanShift = MeanShift(bandwidth=width)
            tempMeanShift.fit(X_train)
            tempPrediction = tempMeanShift.predict(X_train)
            randomIndexes = np.append(randomIndexes, randIndexComputing(y_train, tempPrediction))
    
        return kernelWidths[numpy.argmax(randomIndexes)], randomIndexes


    # find the best k
    bestWidth, arr = tuning(X_train, y_train)
    
    # initialize the model with thew right kernel width
    meanShiftModel = MeanShift(bandwidth=bestWidth)
    
    # clusterization phase
    meanShiftModel.fit(X_train)
    
    # check how many clusters have been inferred
    labels = meanShiftModel.labels_
    cluster_centers = meanShiftModel.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)

"""

"""
    # INTRINSIC EVALUATION
    
    bandwidth = estimate_bandwidth(X_train)

    print("Sk learn best width: ", bandwidth)


    def tuning(X_train) -> tuple:
        randomIndexes: ndarray[float] = np.array([])
        kernelWidths: list[float] = [1.25, 9.5, 30.75, 50.25, 70.5, 100.75, 200.25, 350.5]
        tempList: list = []
        for width in kernelWidths:
            print("width: ", width)
            tempMeanShift = MeanShift(bandwidth=width)
            tempMeanShift.fit(X_train)
            tempPrediction = tempMeanShift.predict(X_train)
            tempList.append(tempPrediction)
    
        for i in range(0, len(tempList) - 1):
            tempMax = 0
            for j in range(i + 1, len(tempList)):
                print("----------------")
                print("fst Width", kernelWidths[i])
                print("sdn Width", kernelWidths[j])
                print("RandIndexes:")
                tempRandIdx = randIndexComputing(tempList[i], tempList[j])
                print("tempRandIndex 1: ", tempRandIdx)
                print("----------------")
                tempRandIdx = 0 if tempRandIdx == 1 else tempRandIdx
                tempMax = tempRandIdx if tempRandIdx > tempMax else tempMax
    
            randomIndexes = np.append(
                randomIndexes,
                tempMax
            )
    
        winningIdx = numpy.argmax(randomIndexes)
    
        return kernelWidths[winningIdx if randomIndexes[winningIdx] >= 0.5 else winningIdx + 1], randomIndexes


    # find the best k
    bestWidth, arr = tuning(X_train)
    
    print("Porcoddio bestwidth: ", bestWidth)
    
    # initialize the model with thew right kernel width
    meanShiftModel = MeanShift(bandwidth=bestWidth)
    
    # clusterization phase
    meanShiftModel.fit(X_train)
    
    # check how many clusters have been inferred
    labels = meanShiftModel.labels_
    cluster_centers = meanShiftModel.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
"""

"""
    Factory implementation
"""


class MeanShiftFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (kernel width)
        self.kernelWidths: list[float] = [1.25, 9.5, 30.75, 50.25, 70.5, 100.75, 200.25, 350.5]

        # generate k mean shift models each of which with a different kernel size
        self.meanShiftModel: list[MeanShift] = [
            MeanShift(
                bandwidth=kernelWidth
            ) for kernelWidth in self.kernelWidths
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(meanShiftModel: MeanShift, X_t: pd.DataFrame):
            # train the model
            meanShiftModel.fit(X_t)

            # return how the model doing with the training data
            return meanShiftModel.predict(X_t)

        # set up the base class
        super().__init__(
            self.meanShiftModel, self.kernelWidths, _fitNdTesting, self._X, self._y
        )

    def tuningProcess(self, returnBestModel: bool = False) -> tuple:
        return super().tuningProcess(returnBestModel)

    def modelsBuilder(self):
        print()
