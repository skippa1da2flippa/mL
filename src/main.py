from src.classifier.gaussian_mixture import GaussianMixturesFactory
from src.utility.fetchedData import X_train, y_train

gaussian = GaussianMixturesFactory(X_train, y_train, 197, 200)

print(gaussian.modelsBuilder())