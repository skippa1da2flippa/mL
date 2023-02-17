from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graphGenerator(pca: PCA):
    # Bar plot of explained_variance
    plt.bar(
        range(1, len(pca.explained_variance_) + 1),
        pca.explained_variance_
    )

    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.show()


class PcaFeatsWrapper:

    # initialize the object with the right bound (interval of dimensions)
    def __init__(self, lowerBound: int = 2, upperBound: int = 200):
        if upperBound > lowerBound > 0:
            self.lowerBound: int = lowerBound
            self.upperBound: int = upperBound
        else:
            self.lowerBound, self.upperBound = 2, 200

    # generate a collection of dataset each of which is reduced to a d-dimensional space
    def pcaIterable(self, dataSet: pd.DataFrame) -> list[pd.DataFrame]:
        pcasContainer: list[pd.DataFrame] = []

        for component in range(self.lowerBound, self.upperBound + 1):
            # initialize PCA
            tempPca: PCA = PCA(n_components=component)

            # fit and transform data
            pca_features = tempPca.fit_transform(dataSet)

            # plot the explained variance graph
            graphGenerator(tempPca)

            # generate a new panda dataframe
            tempPcaDf: pd.DataFrame = pd.DataFrame(
                data=pca_features,
                columns=[col for col in range(0, component)]
            )

            # append the new dataframe
            pcasContainer.append(tempPcaDf)

        return pcasContainer
