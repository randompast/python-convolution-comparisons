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

@njit
def test_2_nbdh(x,k,n): #numba_dot_helper
    r = 0
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            r += x[i+n] * x[j+n] * k[i,j]
    return r

@t
@njit
def test_2_nbconv_wh(x,k): #numba_conv
    l = x.size - k.shape[0]+1
    y = np.zeros(l)
    for i in range(l):
        y[i] = test_2_nbdh(x,k,i)
    return y

@t
@njit
def test_2_nbconv(x,k): #numba_conv
    l = x.size - k.shape[0]+1
    y = np.zeros(l)
    for n in range(l):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                y[n] += x[i+n] * x[j+n] * k[i,j]
    return y


@cuda.jit
def test_2_nbcj(x,k,y): #numba_conv
    for n in range(y.size):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                y[n] += x[i+n] * x[j+n] * k[i,j]

@t
def test_2_nbc(x, k):
    # test_1_nb_cuda_jit.__name__ isn't an attribute
    # this also fixes the api so it can be called with (x,k)
    l = x.size - k.shape[0]+1
    y = cp.zeros(l)
    test_2_nbcj[1,32](x, k, y)
    return y

@t
def test_2_npfe(x,k): #np for loop einsum
    y = np.zeros( x.size )
    for i in range( k.shape[0] - 1 , x.size ):
        X = np.outer(x[1+i-k.shape[0] : 1+i], x[1+i-k.shape[0] : 1+i])
        y[i] = np.einsum( 'ij,ij', X, k)
    return y[k.shape[0]-1:]

@t
def test_2_npe(x,k):
    X = np.asarray( [ np.outer(x[1+i-k.shape[0] : 1+i], x[1+i-k.shape[0] : 1+i]) for i in range( k.shape[0] - 1 , x.size ) ])
    return np.einsum('nij,ij',X,k)

def test_2_valid(n,m):
    # x = np.random.uniform(0,1,n)
    x = np.arange(n)
    k = np.random.uniform(0,1,m)
    K = np.outer( np.arange(m), np.arange(m) )+2

    if True:
        npfe = test_2_npfe(x,K)
        npe = test_2_npe(x,K)
        nbconv = test_2_nbconv(x,K)
        nbc = test_2_nbc(x,K)
        # nbconv_wh = test_2_nbconv_wh(x,K)
        print(npfe)
        print(npe)
        print(nbconv)
        print(nbc)
        print(np.all([
                np.isclose(npfe,npe)
                ,np.isclose(nbconv,npe)
                ,np.isclose(nbc,npe)
                # ,np.isclose(nbconv_wh,npe)
                ])
                , n, m
            )

def test_2_plot():
    for i in range(1,6):
        test_2_valid(4**i, 2**i)

    print(results)
    for k in results:
        a = np.asarray(results[k])
        x = [4**(i+1) * 2**(i+1) for i in range(a.shape[0])]
        # x = [a[i][0]*a[i][1] for i in range(a.shape[0])]
        y = a[:,2]
        # print(k, x, y)
        plt.plot(x, y, label=k[7:])
        plt.text(x[-1], y[-1], str(k[7:]))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # print(test_2_nbconv.__dir__())
    test_2_valid(8, 2)
    test_2_valid(8, 3)
    test_2_valid(8, 4)
    # test_2_valid(1000, 100)
    # test_2_plot()
