import autograd.numpy as np
import os
import string

def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    x = np.random.uniform(-0.5, 0.5, (dim, num))
    dataset = {}
    dataset['train_x'] = x
    dataset['train_y'] = funct(x, bounds)
    return dataset

def get_test(funct, num, bounds):
    dim = bounds.shape[1]
    dataset = {}
    dataset['test_x'] = np.random.uniform(-0.5, 0.5, (dim, num))
    dataset['test_y'] = funct(dataset['test_x'], bounds)
    return dataset

def branin(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    tmp1 = -1.275*np.square(x[0]/np.pi) + 5*x[0]/np.pi + x[1] - 6
    tmp2 = (10 - 5/(4*np.pi))*np.cos(x[0])
    ret = tmp1*tmp1 + tmp2 + 10
    return ret.reshape(1,-1)

def hartmann3d(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*0.0001
    alpha = np.array([1.0,1.2,3.0,3.2])
    ret = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:,i] - P)**2
        tmp = tmp.sum(axis=1)
        ret[0,i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def hartmann6d(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])*0.0001
    ret = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:, i]-P)**2
        tmp = tmp.sum(axis=1)
        ret[0, i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def ellipsoid(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    dim = bounds.shape[0]
    y = ((x**2).T * np.arange(1,dim+1)).T.sum(axis=0)
    return y.reshape(1, -1)

def Dixon_Price(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = (x[0]-1)**2
    for i in range(1,x.shape[0]):
        y = y + (i+1)*(2*x[i]**2 - x[i-1])**2
    return y.reshape(1, -1)

def Styblinski_Tang(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 0.5*(x**4 - 16*x**2 + 5*x).sum(axis=0)
    return y.reshape(1, -1)

def Levy(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    dim = x.shape[0]
    w = 1 + 0.25*(x-1)
    y = np.sin(np.pi*w[0])**2
    y = y + ((w[:dim-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:dim-1]+1)**2)).sum(axis=0)
    y = y + (1 + np.sin(2*np.pi*w[dim-1])**2) * (w[dim-1]-1)**2
    return y.reshape(1, -1)

def Ackley1(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt((x**2).mean(axis=0))) - np.exp(np.cos(2*np.pi*x).mean(axis=0))
    return y.reshape(1, -1)

def Ackley2(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt((x**2).mean(axis=0))) - np.exp(np.cos(2*np.pi*x).mean(axis=0))
    return y.reshape(1, -1)

def get_funct(funct):
    if funct == 'branin':
        return branin
    elif funct == 'hartmann3d':
        return hartmann3d
    elif funct == 'hartmann6d':
        return hartmann6d
    elif funct == 'Ackley1':
        return Ackley1
    elif funct == 'Ackley2':
        return Ackley2
    elif funct == 'ellipsoid':
        return ellipsoid
    elif funct == 'Dixon_Price':
        return Dixon_Price
    elif funct == 'Styblinski_Tang':
        return Styblinski_Tang
    elif funct == 'Levy':
        return Levy
    else:
        return branin
    


