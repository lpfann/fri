import numpy as np

def similarity(u,v):
    u = np.asarray(u)
    v = np.asarray(v)
    diff = u - v
    diff[u == 0] = 0
    diff[v == 0] = 0
    return np.abs(np.sum(diff))/ (np.sum(u)+np.sum(v))

if __name__ == '__main__':
    f1 = [0, 0.2, 0.01]
    f2 = [0.2, 0, 0.01]
    f3 = [0.9, 4, 0]

    print(similarity(f1,f3))