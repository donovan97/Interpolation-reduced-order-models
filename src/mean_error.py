import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from pod import POD
from interpolation import RBFInterpolate
from isomap import Isomap
from pod_nn import main as podNetwork
import matplotlib.pyplot as plt

if __name__ == "__main__":

    A = np.genfromtxt('data/onera_62pts_50.csv', delimiter=',')
    X = A[:-1, 19:] #Exclude bottom row, contains x values
    params = A[:-1, 0:2]
    xvalues = A[-1,19:]

    loo = LeaveOneOut()

    isomap_error = 0
    pod_error = 0
    pod_error_nn = 0

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        params_train, params_test = params[train_index], params[test_index]

        scaler = preprocessing.MinMaxScaler()
        params_train = scaler.fit_transform(params_train) #Scale data points, very important
        params_test = scaler.transform(params_test)

        #Isomap
        isomap = Isomap(X_train, params_train, n_fit=16, n_inv=6)
        y_iso = isomap.transform()
        y_new = RBFInterpolate(params_train, params_test, y_iso)
        W_isomap = isomap.inverse(y_new)
        isomap_error = isomap_error + np.linalg.norm(X_test-W_isomap)

        #POD-standard
        pod = POD(X_train, n_dims=6)
        a_linear = pod.transform()
        a_new_linear = RBFInterpolate(params_train, params_test, a_linear)
        W_pod_linear = pod.inverse(a_new_linear)
        pod_error = pod_error + np.linalg.norm(X_test - W_pod_linear)

        #POD-NN
        snapsBasis, snapsCoeffs = pod.neuralPOD()
        network = podNetwork(inputs=params_train,
                             targets=snapsCoeffs,
                             dimension=6,
                             kfState=False,
                             plots=False)
        W_pod_neural = pod.neuralInverse(network=network, params=params_test)
        pod_error_nn = pod_error_nn + np.linalg.norm(X_test - W_pod_neural)



        fig = plt.figure()
        plt.plot(xvalues[:], X_test[0,:], linewidth=1)
        plt.plot(xvalues[:], W_isomap[0, :], linewidth=1)
        plt.plot(xvalues[:], W_pod_linear[0, :], linewidth=1)
        plt.legend(['CFD solution','Isomap', 'Full POD', 'Targeted POD'],fontsize=14)
        plt.title(scaler.inverse_transform(params_test))
        plt.gca().invert_yaxis()
        params_plt = {'mathtext.default': 'regular'}
        plt.rcParams.update(params_plt)
        plt.xlabel("X", fontsize=14)
        plt.ylabel("$C_p$", fontsize=14)
        plt.show()



    print("Isomap error:", isomap_error/len(X))
    print("POD error:", pod_error/len(X))
    print("POD NN error:", pod_error_nn/len(X))