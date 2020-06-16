import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve as cpxc
from numba import cuda, jit, njit, vectorize
from scipy.signal import convolve as spc
from scipy.signal import fftconvolve as spfc
import time

results = {}

def t(f):
    def wrapper(*args):
        start = time.time()
        for i in range(3):
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
def test_1_naive(x, k):
    y = np.zeros( x.size )
    for i in range( k.size - 1 , x.size ):
        for j in range( k.size ):
            y[i] += x[i-j] * k[j]
    return y[k.size - 1:]

@t
def test_1_npc(x, k, m):
    return np.convolve(x, k, m)

@t
def test_1_npfe(x, k):
    y = np.zeros( x.size )
    # for i in range( k.size, x.size+1 ):
    #     y[i-1] = np.einsum('i,i', x[i-k.size:i], k[::-1])
    for i in range( k.size-1, x.size ):
        y[i] = np.einsum('i,i', x[1+i-k.size : 1+i], k[::-1])
    return y[k.size-1:]

@t
def test_1_npste(x, k):
    return np.einsum('ni,i', x, k[::-1])

@t
def test_1_cpste(x, k):
    return cp.einsum('ni,i', x, k[::-1])

@t
def test_1_cpv(x, k):
    return cpxc(x, k)[ k.size/2 + k.size%2 -1 : -k.size/2 ]

@t
def test_1_cp_same(x, k):
    return cpxc(x, k, mode='constant', origin=(k.size%2-1))

@t
def test_1_cp_full(x, k):
    return cpxc(x, k, mode='constant')[ k.size/2 + k.size%2 : -k.size/2 -1]

@t
def test_1_spcv(x, k):
    return spc(x, k, 'valid')

@t
def test_1_spfcv(x, k): #fftconvolve
    return spfc(x, k, 'valid')

@t
def test_1_npd(x, k):
    y = np.zeros(x.size)
    for i in range(k.size-1,x.size):
        y[i] = np.dot(x[1+i-k.size : 1+i][::-1],k)
    return y[k.size - 1:]

# @vectorize(['float64[:](float64[:],float64[:])'], target='cpu')
@t
@njit(parallel=True)
def test_1_nbp(x, k):
    y = np.zeros( x.size )
    for i in range( k.size - 1 , x.size ):
        for j in range( k.size ):
            y[i] += x[i-j] * k[j]
    return y[k.size - 1:]

@t
@njit(parallel=True)
def test_1_nbpd(x, k):
    y = np.zeros( x.size )
    for i in range( k.size -1 , x.size ):
        y[i] = np.dot(x[1+i-k.size : 1+i][::-1],k)
    return y[k.size-1:]

@t
@njit(nogil=True, parallel=True)
def test_1_nbpd_ng(x, k):
    y = np.zeros( x.size )
    for i in range( k.size -1 , x.size ):
        for j in range( k.size ):
            y[i] += x[i-j] * k[j]
    return y[k.size-1:]

@t
@jit
def test_1_nbnpc(x, k):
    return np.convolve(x,k)[k.size-1:x.size]

@t
@njit
def test_1_nbnpc_nj(x, k):
    return np.convolve(x,k)[k.size-1:x.size]


@cuda.jit
def test_1_nb_cuda_jit(x, k, y):
    # y = np.zeros(x.size) #doesn't work
    for i in range( k.size -1 , x.size ):
        for j in range( k.size ):
            y[i-k.size+1] += x[i-j] * k[j]
    # return y #doesn't work

@cuda.jit
def test_1_nb_cuda_jit_gmarkall(x, k, y):
    i = cuda.grid(1)
    if (i >= k.size - 1) and (i < x.size):
        for j in range( k.size ):
            y[i-k.size+1] += x[i-j] * k[j]
    # return y #doesn't work


@t
def test_1_nbc(x, k):
    # test_1_nb_cuda_jit.__name__ isn't an attribute
    # this also fixes the api so it can be called with (x,k)
    y = cp.zeros(x.size-k.size+1)
    test_1_nb_cuda_jit[1,32](x, k, y)
    # test_1_nb_cuda_jit[128,128](x, k, y) #fails tests
    return y

@t
def test_1_nbc_gm(x, k):
    y = cp.zeros(x.size-k.size+1)
    test_1_nb_cuda_jit_gmarkall[x.size // 128  + 1 , 128](x, k, y)
    return y


def test_1_valid(n,m):
    x = np.random.uniform(0,1,n)
    k = np.random.uniform(0,1,m)
    xc = cp.asarray(x)
    kc = cp.asarray(k)
    xs = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0] - k.shape[0] + 1, )
        + k.shape, strides=x.strides[:1] + x.strides)
    xcs = cp.lib.stride_tricks.as_strided(xc, shape=(xc.shape[0] - kc.shape[0] + 1, )
        + kc.shape, strides=xc.strides[:1] + xc.strides)

    # a_naive = test_1_naive(x,k)#[k.size:]
    a_npv = test_1_npc(x,k,'valid')#[:x.size-k.size]
    a_npfe = test_1_npfe(x, k)
    a_npste = test_1_npste(xs, k)
    a_cpste = test_1_cpste(xcs, kc)
    a_cpv = test_1_cpv(cp.asarray(x),cp.asarray(k)) #[1:x.size-k.size+1]
    a_spcv = test_1_spcv(x,k)#[:x.size-k.size]
    a_spfcv = test_1_spfcv(x,k)#[:x.size-k.size]
    a_npd = test_1_npd(x,k)
    a_nbp = test_1_nbp(x,k)
    # a_nbpd = test_1_nbpd(x,k)
    a_nbpd_ng = test_1_nbpd_ng(x, k)
    a_nbnpc = test_1_nbnpc(x,k)
    a_nbnpc_nj = test_1_nbnpc_nj(x,k)
    a_nbc = test_1_nbc(xc, kc)
    a_nbc_gm = test_1_nbc_gm(xc, kc)

    # print("a_naive", a_naive)
    # print("a_npv", a_npv)
    # print("a_npfe", a_npfe)
    # print("a_npste", a_npste)
    # print("a_cpste", a_cpste)
    # print("a_cpv", a_cpv)
    # print("a_spcv", a_spcv)
    # print("a_spfcv", a_spfcv)
    # print("a_npd", a_npd)
    # print("a_nbp", a_nbp)
    # print("a_nbpd", a_nbpd)
    # print("a_nbpd_ng", a_nbpd_ng)
    # print("a_nbnpc", a_nbnpc)
    # print("a_nbnpc_nj", a_nbnpc_nj)
    # print("a_nbc", a_nbc)

    print(np.all([
        # np.all(np.isclose(a_naive, a_npv)),
        np.all(np.isclose(a_npv, a_npv))
        ,np.all(np.isclose(a_npfe, a_npv))
        ,np.all(np.isclose(a_npste, a_npv))
        ,np.all(np.isclose(a_cpste, a_npv))
        ,np.all(np.isclose(a_cpv, a_npv))
        ,np.all(np.isclose(a_spcv, a_npv))
        ,np.all(np.isclose(a_spfcv, a_npv))
        ,np.all(np.isclose(a_npd, a_npv))
        ,np.all(np.isclose(a_nbp, a_npv))
        # ,np.all(np.isclose(a_nbpd, a_npv))
        ,np.all(np.isclose(a_nbpd_ng, a_npv))
        ,np.all(np.isclose(a_nbnpc, a_npv))
        ,np.all(np.isclose(a_nbnpc_nj, a_npv))
        ,np.all(np.isclose(a_nbc, a_npv))
        ,np.all(np.isclose(a_nbc_gm, a_npv))
        ]))

def test_1_all(n,m):
    x = np.random.uniform(0,1,n)
    k = np.random.uniform(0,1,m)
    a_npv = test_1_npc(x,k,'valid')#[:x.size-k.size]
    a_nps = test_1_npc(x,k,'same')#[:x.size-k.size]
    a_npf = test_1_npc(x,k,'full')#[:x.size-k.size]
    a_cpv = test_1_cpv(cp.asarray(x),cp.asarray(k)) #[1:x.size-k.size+1]
    a_cps = test_1_cp_same(cp.asarray(x),cp.asarray(k)) #[1:x.size-k.size+1]
    a_cpf = test_1_cp_full(np.concatenate((cp.zeros(k.size), cp.asarray(x), cp.zeros(k.size))),cp.asarray(k)) #[1:x.size-k.size+1]
    a_spc = test_1_spcv(x,k)#[:x.size-k.size]

    print(  np.all( np.isclose(a_npv, a_cpv) )  )
    print(  np.all( np.isclose(a_nps, a_cps) )  )
    print(  np.all( np.isclose(a_npf, a_cpf) )  )
    print(  np.all( np.isclose(a_npv, a_spc) )  )

def test_1_scrap():
    # test_1_valid(10,2)
    # test_1_valid(10,3)
    # test_1_valid(10,4)
    test_1_all(10,2)
    test_1_all(10,3)
    test_1_all(10,4)

def test_1_plot():
    for i in range(1,9):
        test_1_valid(4**i, 2**i)

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
    # print(test_1_nbc.__dir__())
    # test_1_valid(8, 4)
    # test_1_valid(1000, 100)
    # test_1_valid(20*1000, 20*100)
    test_1_plot()
