import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import random


class SupportVectorMachine:
    def __init__(self, data, labels, dimension, n):
        self.data = data
        self.label = labels
        self.dimension = dimension
        self.data_points = n
        self.lin_sep_sol = None
        self.lin_sep_weights = np.zeros((self.data_points, 1))
        self.lin_sep_bias = None

    def get_P(self, kernel):
        if not kernel:
            P = self.data * self.label
            P = np.dot(P, P.T)
            return matrix(P)
        else:
            P = np.zeros((self.data_points, self.data_points))
            for i in range(self.data_points):
                for j in range(self.data_points):
                    P[i, j] = self.kernel(i, j)
            # P = np.multiply()

    def weights(self):
        print(np.multiply(self.lin_sep_sol, np.multiply(self.label, self.data)))
        self.lin_sep_weights = np.sum(np.multiply(self.lin_sep_sol, np.multiply(self.label, self.data)), axis=0)
        print("Weights:")
        print(self.lin_sep_weights)

    def calc_bias(self, support_vectors):
        random_index = random.randint(0, len(support_vectors)-1)
        random_index = support_vectors[random_index]
        self.lin_sep_bias = (1/self.label[random_index]) - (np.dot(self.lin_sep_weights.T, self.data[random_index]))
        print("Bias:")
        print(self.lin_sep_bias)

    def train(self):
        print(self.data.shape)
        P = self.get_P(False)
        print(self.data_points)
        q = matrix(np.ones(self.data_points) * -1)
        b = matrix(0.0)
        h = matrix(np.zeros((self.data_points, 1)))
        a = self.label.T.astype('float')
        A = matrix(a)
        G = matrix(np.diag(np.ones(self.data_points) * -1))
        solution_ = solvers.qp(P, q, G, h, A, b)
        solution = np.array(solution_['x'])
        solution[solution < 0.00001] = 0
        self.lin_sep_sol = solution
        support_vectors = np.where(self.lin_sep_sol != 0)[0]
        print("Support Vector Indices", support_vectors)
        self.weights()
        self.calc_bias(support_vectors)
        self.predict()

    def predict(self):
        prediction = 0
        misclassified = 0
        for i in range(self.data_points):
            signal = np.dot(self.lin_sep_weights.T, self.data[i]) + self.lin_sep_bias
            if signal > 0:
                prediction = 1
            else:
                prediction = -1
            if prediction != self.label[i]:
                misclassified += 1
            # print("Misclassified", misclassified)
        print("Accuracy - Linear separable", 1-(misclassified//self.data_points))


class SupportVectorMachineNS:
    def __init__(self, data, labels, dimension, n):
        self.data = data
        self.label = labels
        self.dimension = dimension
        self.data_points = n
        self.non_lin_sep_sol = None
        self.non_lin_sep_weights = np.zeros((self.data_points, 1))
        self.non_lin_sep_bias = None
        self.P = np.zeros((self.data_points, self.data_points))

    def kernel(self, i, j):
        return (1 + np.dot(self.data[i], self.data[j].T))**3

    def get_P(self, kernel):
        if not kernel:
            P = self.data * self.label
            P = np.dot(P, P.T)
            return matrix(P)
        else:
            print("From here")
            for i in range(self.data_points):
                for j in range(self.data_points):
                    self.P[i, j] = self.kernel(i, j)
            y = np.dot(self.label, self.label.T)
            self.P = y*self.P
            return matrix(self.P)

    def predict(self, support_vectors):
        print(self.non_lin_sep_sol[support_vectors])
        prediction = 0
        misclassified = 0
        for i in range(self.data_points):
            val = 0
            for support_vector in support_vectors:
                val += self.non_lin_sep_sol[support_vector] * self.label[support_vector] * self.kernel(support_vector, i)
            if (val + self.non_lin_sep_bias) >= 0:
                prediction = 1
            else:
                prediction = -1
            if prediction != self.label[i]:
                misclassified += 1
        print("Accuracy:", (1-(misclassified/self.data_points)))

    def calc_bias(self, support_vectors):
        # random_index = random.randint(0, len(support_vectors)-1)
        # random_index = support_vectors[random_index]
        random_index = support_vectors[2]
        val = 0
        for support_vector in support_vectors:
            val += self.non_lin_sep_sol[support_vector] * self.label[support_vector] * self.kernel(random_index, support_vector)
        self.non_lin_sep_bias = self.label[random_index] - val
        print("Bias:")
        print(self.non_lin_sep_bias)

    def train(self):
        print(self.data.shape)
        self.P = self.get_P(True)
        q = matrix(np.ones(self.data_points) * -1)
        b = matrix(0.0)
        h = matrix(np.zeros((self.data_points, 1)))
        a = self.label.T.astype('float')
        A = matrix(a)
        G = matrix(np.diag(np.ones(self.data_points) * -1))
        solution_ = solvers.qp(self.P, q, G, h, A, b)
        solution = np.array(solution_['x'])
        print(solution)
        solution[solution <= 1e-4] = 0
        self.non_lin_sep_sol = solution
        support_vectors = np.where(self.non_lin_sep_sol != 0)[0]
        print("Support Vector Indices", support_vectors)
        self.calc_bias(support_vectors)
        self.predict(support_vectors)


if __name__ == "__main__":
    print("Reading the data - Separable")
    dataframe = pd.read_csv('linsep.txt', names=['x', 'y', 'label'])
    data_ = dataframe[['x', 'y']].copy().to_numpy()
    label = dataframe[['label']].copy().to_numpy()
    svm = SupportVectorMachine(data_, label, data_.shape[1], data_.shape[0])
    svm.train()
    print("=========================================================================================================")
    print("Reading the data - Non Separable")
    dataframe_ns = pd.read_csv('nonlinsep.txt', names=['x', 'y', 'label'])
    data_ns = dataframe_ns[['x', 'y']].copy().to_numpy()
    label_ns = dataframe_ns[['label']].copy().to_numpy()
    svm_ns = SupportVectorMachineNS(data_ns, label_ns, data_ns.shape[1], data_ns.shape[0])
    svm_ns.train()

