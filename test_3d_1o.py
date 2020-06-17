import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import cuda, jit, njit, vectorize
from cupyx.scipy.ndimage import convolve as cpxc
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
def test_3d1o_nbconv(x,k): #numba_conv
    l = x.shape[0] - k.shape[0] + 1
    y = np.zeros(l)
    for n in range(l):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x[i+n,j,l] * k[i,j,l]
    return y

@t
def test_3d1o_npste(x,k):
    xw = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0] - k.shape[0] + 1, )
        + k.shape, strides=x.strides[:1] + x.strides)
    return np.einsum('nijk,ijk', xw, k)

@t
def test_3d1o_cpste(x,k):
    xw = cp.lib.stride_tricks.as_strided(x, shape=(x.shape[0] - k.shape[0] + 1, )
        + k.shape, strides=x.strides[:1] + x.strides)
    return cp.einsum('nijk,ijk', xw, k)

@t
def test_3d1o_cpv(x, k):
    return cpxc(x, k)[ k.shape[0]/2 + k.shape[0]%2 -1 : -k.shape[0]/2
        , k.shape[1]/2 + k.shape[1]%2 -1
        , k.shape[2]/2 + k.shape[2]%2 -1 ]

@t
def test_3d1o_spv(x, k):
    return spc(x, k, mode='valid').flatten()

@cuda.jit
def test_3d1o_nbcj_grid(x,k,y): #numba_conv
    n = cuda.grid(1)
    if (0 <= n) and (n < y.size):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                for l in range(k.shape[2]):
                    y[n] += x[i+n,j,l] * k[i,j,l]

@t
def test_3d1o_nbcg(x, k):
    l = x.shape[0] - k.shape[0] + 1
    y = cp.zeros(l)
    th = 128
    b = y.size//th+1
    # print(b,th)
    test_3d1o_nbcj_grid[b,th](x, k, y)
    return y

def test_3d1o_valid(n,m):
    np.random.seed(0)
    x = np.random.uniform(-1,1,(n,m,m))
    k = np.random.uniform(-1,1,(m,m,m))
    # x = np.arange(n*m*m).reshape(n,m,m)
    # k = np.arange(m*m*m).reshape(m,m,m)
    # kinv = np.arange(m*m*m)[::-1].reshape(m,m,m)
    kinv = k.flatten()[::-1].reshape(m,m,m)
    xc = cuda.to_device(x)
    kc = cuda.to_device(k)
    xcp = cp.asarray(x)
    kcp = cp.asarray(k)
    kcpinv = cp.asarray(kinv)

    # print(x)
    # print(k)

    if True:
        nbconv = test_3d1o_nbconv(x, k)
        nbcg = test_3d1o_nbcg(xc, kc)
        npste = test_3d1o_npste(x, k)
        cpste = test_3d1o_cpste(xcp, kcp)
        spv = test_3d1o_spv(x, kinv)
        cpv = test_3d1o_cpv(xcp, kcpinv)

        # print(nbconv)
        # print(npste)
        # print(cpste)
        # print(nbcg)
        # print(spv)
        # print(cpv)

        print(np.all([
                np.isclose(nbcg,nbconv)
                ,np.isclose(npste,nbconv)
                ,np.isclose(cpste,nbconv)
                ,np.isclose(spv,nbconv)
                ,np.isclose(cpv,nbconv)
                ])
                , n, m
            )

def test_3d1o_plot():
    for i in range(1,6):
        test_3d1o_valid(4**i, 2**i)

    print(results)
    for k in results:
        a = np.asarray(results[k])
        x = [4**(i+1) * 2**(5*(i+1)) for i in range(a.shape[0])]
        y = a[:,2]
        plt.plot(x[1:], y[1:], label=k[10:])
        plt.text(x[-1], y[-1], str(k[10:]))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_3d1o_valid(5, 2)
    # test_3d1o_valid(5, 3)
    # test_3d1o_valid(5, 4)
    test_3d1o_plot()
