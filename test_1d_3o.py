import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import cuda, jit, njit, vectorize
# from numba import njit
from scipy.signal import convolve as spc
from scipy.signal import fftconvolve as spfc
import time

results = {}

def t(f):
    def wrapper(*args):
        start = time.time()
        for i in range(10):
            a = f(*args)
        print(f.__name__, time.time()-start)
        data = [[args[0].size, args[1].size, time.time()-start]]
        if f.__name__ in results:
            results[f.__name__] += data
        else:
            results[f.__name__] = data
        return a
    return wrapper

@t
@njit
def test_3_nbconv(x,k): #numba_conv
    l = x.size - k.shape[0]
    y = np.zeros(l)
    for n in range(l):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x[i+n] * x[j+n] * x[l+n] * k[i,j,l]
    return y


@cuda.jit
def test_3_nbcj(x,k,y): #numba_conv
    for n in range(y.size):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x[i+n] * x[j+n] * x[l+n] * k[i,j,l]

@cuda.jit
def test_3_nbcj_grid(x,k,y): #numba_conv
    n = cuda.grid(1)
    if (0 <= n) and (n < y.size):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x[i+n] * x[j+n] * x[l+n] * k[i,j,l]

@t
def test_3_nbc(x, k):
    l = x.size - k.shape[0]
    y = cp.zeros(l)
    test_3_nbcj[1,32](x, k, y)
    return y

@t
def test_3_nbcg(x, k):
    l = x.size - k.shape[0]
    y = cp.zeros(l)
    th = 128
    b = y.size//th+1
    # print(b,th)
    test_3_nbcj_grid[b,th](x, k, y)
    return y

@t
def test_3_nbcg1024(x, k):
    l = x.size - k.shape[0]
    y = cp.zeros(l)
    th = 1024
    b = y.size//th+1
    # print(b,th)
    test_3_nbcj_grid[b,th](x, k, y)
    return y

def test_3_valid(n,m):
    np.random.seed(0)
    x = np.random.uniform(-1,1,n)
    k = np.random.uniform(-1,1,(m,m,m))

    if True:
        nbconv = test_3_nbconv(x,k)
        # nbc = test_3_nbc(x,k)
        nbcg = test_3_nbcg(x,k)
        nbcg1024 = test_3_nbcg1024(x,k)

        # print(nbconv)
        # print(nbc)
        # print(nbcg)

        print(np.all([
                np.isclose(nbconv,nbconv)
                # ,np.isclose(nbc,nbconv)
                ,np.isclose(nbcg,nbconv)
                ,np.isclose(nbcg1024,nbconv)
                ])
                , n, m
            )

def test_3_plot():
    for i in range(1,6):
        test_3_valid(4**i, 2**i)

    print(results)
    for k in results:
        a = np.asarray(results[k])
        x = [4**(i+1) * 2**(i+1) for i in range(a.shape[0])]
        y = a[:,2]
        plt.plot(x[1:], y[1:], label=k[7:])
        plt.text(x[-1], y[-1], str(k[7:]))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_3_valid(7, 2)
    # test_3_valid(7, 3)
    # test_3_valid(7, 4)
    # test_3_valid(17, 12)
    # test_3_valid(17, 13)
    # test_3_valid(17, 14)
    # test_3_valid(64, 8)
    test_3_plot()
