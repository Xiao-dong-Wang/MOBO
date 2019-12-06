import autograd.numpy as np
import sys
import toml
from src.util import *
from src.BO import BO
from get_dataset import *
import multiprocessing
import pickle

def f1(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    return x[0,:].reshape(1,-1)

def f2(x, bounds):
    tmp = f1(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    tmp_g = 1 + 9 * np.sum(x[1:,:],0)
    tmp_h = 1 - np.sqrt(tmp / tmp_g)
    return (tmp_g * tmp_h).reshape(1,-1)


#data = init_dataset(funct, num, bounds)
dim = 6
outdim = 2
num_obj = 2
num = 10
bounds = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]])

x = np.random.uniform(-0.5, 0.5, (dim, num))
y = np.zeros((outdim, num))
y[0] = f1(x,bounds)
y[1] = f2(x,bounds)
dataset = {}
dataset['origin_x'] = x
dataset['origin_y'] = y

new_bounds = np.concatenate((bounds, np.array([[-0.5, 0.5]]))) 
bfgs_iter = np.array([2000, 2000, 2000])
K = 100
iteration = 200

with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)


for ii in range(iteration):
    print('********************************************************************')
    print('iteration',ii)

    weight = np.random.uniform(0, 1, (num_obj))
    weight = weight / np.sum(weight)
    origin_x = dataset['origin_x']
    origin_y = dataset['origin_y']

    dim, n = np.shape(origin_x)
    train_x = np.zeros((dim+1, n))
    train_x[:dim, :] = origin_x
    train_y = np.zeros(( 1+outdim, n)) 
    for i in range(origin_x.shape[1]):
        lamd = np.random.uniform(-0.5, 0.5, (1))
        train_x[dim, i] = lamd
        train_y[0, i] = lamd
        train_y[1:num_obj+1,i] = weight * origin_y[:num_obj,i] - lamd
        train_y[num_obj+1:,i] = origin_y[num_obj:,i]

    dataset['train_x'] = train_x
    dataset['train_y'] = train_y
    model = BO(dataset, new_bounds, bfgs_iter, debug=False)
    best_x = model.best_x
    best_y = model.best_y
    print('best_x', best_x)
    print('best_y', best_y)

    
    p = np.minimum(int(K/5), 5)
    def task(x0):
        x0 = model.optimize_constr(x0)
        x0 = model.optimize_wEI(x0)
        wEI_tmp = model.calc_log_wEI_approx(x0)
        return x0, wEI_tmp
    pool = multiprocessing.Pool(processes=5)
    x0_list = []
    for j in range(int(K/p)):
        x0_list.append(model.rand_x(p))
    results = pool.map(task, x0_list)
    pool.close()
    pool.join()

    candidate = results[0][0]
    wEI_tmp = results[0][1]
    for k in range(1, int(K/p)):
        candidate = np.concatenate((candidate.T, results[k][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp.T, results[k][1].T)).T

    idx = np.argsort(wEI_tmp)[-1:]
    new_xl = candidate[:, idx]
    print('new_xl', new_xl)
    new_x = new_xl[:dim]
    new_y1 = f1(new_x, bounds)
    new_y2 = f2(new_x, bounds)
    new_y = np.concatenate((new_y1, new_y2))
    print('new_x', new_x)
    print('new_y1', new_y1)
    print('new_y2', new_y2)

    dataset['origin_x'] = np.concatenate((dataset['origin_x'].T, new_x.T)).T
    dataset['origin_y'] = np.concatenate((dataset['origin_y'].T, new_y.T)).T
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)



