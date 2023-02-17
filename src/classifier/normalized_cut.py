import pandas as pd
from numpy import ndarray
from sklearn.cluster import SpectralClustering

from base_classes import BaseModelFactory

"""
    # EXTRINSIC EVALUATION
    def normalizedCutTuning(X, y) -> tuple:
        randomIndexes: ndarray[float] = np.array([])
        nClusters: list[int] = [2, 5, 9, 20, 35, 55, 75, 100, 150]
    
        for nCluster in nClusters:
            tempSpectralClustering = SpectralClustering(n_clusters=nCluster)
            tempPrediction = tempSpectralClustering.fit_predict(X)
            randomIndexes = np.append(randomIndexes, randIndexComputing(y, tempPrediction))
    
        return nClusters[np.argmax(randomIndexes)], randomIndexes
    
    
    # find the best k
    nCluster, arr = normalizedCutTuning(X_train, y_train)
    print("nCluster: ", nCluster)
    print("random Indexes: ", arr)
    
    # initialize the model with thew right number of clusters
    spectralModel = SpectralClustering(n_clusters=nCluster)
    
    # clusterization phase
    prediction = spectralModel.fit_predict(X_train)
    
    # check how the model is doing
    print(randIndexComputing(y_train, prediction))
"""


"""
    # INTRINSIC EVALUATION

    def normalizedCutTuning(X) -> tuple:
        randomIndexes: ndarray[float] = np.array([])
        nClusters: list[int] = [2, 5, 9, 20, 35, 55, 75, 100, 150]
        predictions: list = []
        for nCluster in nClusters:
            tempSpectralClustering = SpectralClustering(n_clusters=nCluster)
            predictions.append(tempSpectralClustering.fit_predict(X))
    
        for idx in range(0, len(predictions) - 1):
            tempMax = 0
            for idj in range(idx + 1, len(predictions)):
                tempRandIdx = randIndexComputing(predictions[idx], predictions[idj])
                tempMax = tempRandIdx if tempRandIdx > tempMax else tempMax
    
            randomIndexes = np.append(randomIndexes, tempMax)
    
        return nClusters[np.argmax(randomIndexes)], randomIndexes
    
    
    # find the best k
    nCluster, arr = normalizedCutTuning(X_train)
    print("nCluster: ", nCluster)
    print("random Indexes: ", arr)
    
    # initialize the model with thew right number of clusters
    spectralModel = SpectralClustering(n_clusters=nCluster)
    
    # clusterization phase
    prediction = spectralModel.fit_predict(X_train)
    
    # check how the model is doing
    print(randIndexComputing(y_train, prediction))
"""


"""
    Factory implementation
"""


class NormalizedCutFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (number of clusters)
        self.nClusters: list[int] = [2, 5, 9, 20, 35, 55, 75, 100, 150]

        # generate k normalized cut models each of which with a different number of clusters
        self.meanShiftModel: list[SpectralClustering] = [
            SpectralClustering(
                n_clusters=nCluster
            ) for nCluster in self.nClusters
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(normalizedCutModel: SpectralClustering, X_t: pd.DataFrame):
            # train the model and return how the model doing with the training data
            return normalizedCutModel.fit_predict(X_t)

        # set up the base class
        super().__init__(
            self.meanShiftModel, self.nClusters, _fitNdTesting, self._X, self._y
        )

    def tuningProcess(self, returnBestModel: bool = False) -> tuple:
        return super().tuningProcess(returnBestModel)

    def modelsBuilder(self):
        print()

