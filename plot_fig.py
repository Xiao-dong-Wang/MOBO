import autograd.numpy as np
import toml
import pickle
import matplotlib.pyplot as plt

def dominate(obj1, obj2):
    bool =  (obj1 <= obj2).all() and (obj1 < obj2).any()
    return bool

def pareto_calc(X, Y):
    num_dim = X.shape[0]
    num_obj = Y.shape[0]
    num_samples = Y.shape[1]
    tmp_pareto_set = []
    tmp_pareto_front = []
    for i in range(num_samples):
        dominated = False
        for j in range(num_samples):
            if (i != j and (dominate(Y[:,j], Y[:,i]))):
                dominated = True
                break
        if (not dominated):
            tmp_pareto_set.append(X[:,i])
            tmp_pareto_front.append(Y[:,i])

    num_dom = len(tmp_pareto_set)
    pareto_set = np.zeros((num_dim, num_dom))
    pareto_front = np.zeros((num_obj, num_dom))
    for i in range(num_dom):
        pareto_set[:,i] = tmp_pareto_set[i]
        pareto_front[:,i] = tmp_pareto_front[i]

    return pareto_set, pareto_front


with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
sample_x = dataset['origin_x']
sample_y = dataset['origin_y']

PS, PF = pareto_calc(sample_x, sample_y)

y1 = np.linspace(0,1,1000)
y2 = 1- np.sqrt(y1)

plt.figure()
plt.rc('font', family='serif', size=12)
plt.plot(PF[0], PF[1], 'bo', markersize=4.5,label='Sampled points')
plt.plot(y1, y2, 'r-', linewidth=2, label='True Pareto front')
plt.legend()
plt.xlabel('f1')
plt.ylabel('f2')

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([0, 3])

plt.show()
