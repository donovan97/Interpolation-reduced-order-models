import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.utils import graph_shortest_path
from interpolation import RBFInterpolate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import copy

class Isomap:

    def __init__(self, X, params, n_fit = 6, n_inv = 6, n_components = 2):
        #Scale data between [0,1] before applying Isomap
        self.X_scaler = preprocessing.MinMaxScaler()
        self.X_scaled = self.X_scaler.fit_transform(X)
        self.params_scaler = preprocessing.MinMaxScaler()
        self.params = self.params_scaler.fit_transform(params)
        self.n_fit = n_fit
        self.n_inv = n_inv
        self.n_components = n_components


    def transform(self):
        #Compute neighborhood graph
        graph = neighbors.kneighbors_graph(self.X_scaled, self.n_fit, mode='distance')

        #Use Floyd-Warshall algorithm
        self.dist_matrix = graph_shortest_path.graph_shortest_path(graph, method="FW", directed=False)

        #Using classical MDS
        m,n = np.shape(self.dist_matrix)
        C = np.subtract(np.identity(n), (1/n)*np.ones((n,n)))
        B = -0.5*(C @ self.dist_matrix**2 @ C)
        w,v = np.linalg.eig(B)
        indices = np.argsort(w)
        #Ensure no complex values
        w = np.real(w[indices[::-1]])
        v = np.real(v[:,indices[::-1]])
        #Use n_components
        self.y_mapped= v[:,0:self.n_components] @ np.diag(np.sqrt(w[0:self.n_components]))

        return self.y_mapped

    def inverse(self, y_test):
        m, n = np.shape(self.X_scaled)
        W = np.array([]).reshape(0, n)
        for y_new in y_test:
            #Get nearest neighbors and distances
            knn = neighbors.NearestNeighbors(n_neighbors=self.n_inv)
            knn.fit(self.y_mapped)
            distances, indices = knn.kneighbors(y_new.reshape(1,-1))
            G = np.ones((self.n_inv + 1, self.n_inv + 1))

            #Build Gram matrix
            for i in np.arange(self.n_inv):
                for j in np.arange(self.n_inv):
                    G[i,j] = np.dot(np.subtract(y_new.reshape(1,-1), self.y_mapped[indices[0, i],:]), np.subtract(y_new.reshape(1,-1), self.y_mapped[indices[0, j],:]).T)
            G[self.n_inv, self.n_inv] = 0

            #Formula from TFranz report
            c = 0.01 * (distances / np.max(distances)) ** 4
            c = np.append(c, 0)
            cdiag = np.diagflat(c)
            G = G + cdiag
            b = np.zeros((1, self.n_inv))
            b = np.append(b,1)

            #Solve for back mapping weights
            a = np.linalg.solve(G, b)

            #Compute high order inverse mapping
            W_scaled = np.zeros((1,n))
            for k in np.arange(self.n_inv):
                W_scaled = W_scaled + (a[k]*self.X_scaled[indices[0,k]]).reshape(1,-1)

            #Un-scale points
            W_new = self.X_scaler.inverse_transform(W_scaled)
            W = np.concatenate((W, W_new))
        return W

