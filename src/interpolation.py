from scipy.interpolate import Rbf

def RBFInterpolate(points, newpoints, y_manifold, function='thin-plate'):
    rbf = Rbf(points[:, 0], points[:, 1], y_manifold, mode='N-D', function=function)
    y_predict = rbf(newpoints[:,0], newpoints[:, 1])
    return y_predict
