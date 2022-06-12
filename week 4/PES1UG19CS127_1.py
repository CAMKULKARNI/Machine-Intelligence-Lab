import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        if self.p != 0:
            return np.array([np.array([np.power(np.sum(np.power(abs(i - j), self.p)), 1/self.p) for j in self.data]) for i in x])
        else:
            return np.array([np.array([1 for _ in self.data]) for __ in x])

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        distances = self.find_distance(x)
        return [[np.sort(i)[:self.k_neigh] for i in distances], [
            np.argsort(i)[:self.k_neigh] for i in distances]]

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        k_neighbours = self.k_neighbours(x)
        pred = []
        indices = k_neighbours[1]
        for i in range(len(indices)):
            results = {}
            for j in range(len(indices[i])):
                t = self.target[indices[i][j]]
                if self.weighted:
                    de = k_neighbours[0][i][j]
                    de = de if de != 0 else 10e-8
                    if t in results:
                        results[t] += 1 / de
                    else:
                        results[t] = 1 / de
                else:
                    if t in results:
                        results[t] += 1
                    else:
                        results[t] = 1

            max_value = 0
            min_key = float('inf')
            for j in results:
                if results[j] >= max_value and min_key >= j:
                    max_value = results[j]
                    min_key = j
            pred.append(min_key)

        return np.array(pred)

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        return 100 * (sum(self.predict(x) == y) / len(y))
