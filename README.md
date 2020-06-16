# Python Convolution Comparisons

Convolutions are a fundamental operation in scientific computing.  The goal here is to explore the possible approaches in python.  A convolution is a sliding dot product.

Ideal implementation would have the inputs (x,K) where x is 1d or 3d K is [0th order, 1st order, 2nd, order, ... nth order].

Example usages:

    x1 = np.random.uniform(-1,1,100)
    x3 = np.random.uniform(-1,1,(100,10,10))
    k1 = np.random.uniform(-1,1,10))
    k2 = np.random.uniform(-1,1,(10,10))
    k3 = np.random.uniform(-1,1,(10,10,10))
    conv( x3, k3 )
    conv( x1, k1 )
    conv( x1, k2 )
    conv( x1, k3 )
    conv( x1, [ k1 , k2 , k3 ] )

# 1D First Order
An example 1D signal could be audio amplitudes.  An example filter would be an impulse response.  The filter slides over a window of the larger audio signal.  Corresponding terms are multiplied and then summed together to construct a filtered version of the signal.

It was found that numpy.convolve is the fastest for a large variety of inputs, but after a sufficiently large size it makes sense to use cupy.  The gpu implementations started to become worth using at around 10^7.

![1D, First Order](1d1o.png)

Best to worst for larger arrays:
*  cpste - cupy strided [@Alexer](https://github.com/alexer)
*  nbc_gm, nbc - numba cuda improvement from [@gmarkall](https://github.com/gmarkall) [via](https://numba.discourse.group/t/numba-convolutions/33/2)
*  cpv, spfcv, spvc, npc
    *  cpv - cupyx.scipy.ndimage,
    *  spvc - scipy.signal.convolve
    *  spfvc - scipy.signal.fftconvolve
    *  npc - numpy.convolve
*  npste, nbpd_np, nbp, nbnpc, nbnpc_nj
    *  npste - numpy strided [@Alexer](https://github.com/alexer)
    *  nbpd_np - numba parallel dot, no gil
    *  nbp - numba parallel
    *  nbnpc_nj - @njit numba numpy convolve
*  npd - numpy dot
*  npfe - numpy for einsum
*  naive - (not shown) two for loops, significantly worse

# 1D Second Order

![1D, Second Order](1d2o.png)

Best to worst for larger arrays:
*  nbconv - numba njit (naive implementation with njit)
*  npe - numpy einsum (generate the outer product all at once first)
*  npfe - numpy for loop einsum (generate the outer product on demand)
*  nbc - numba cuda (naive implementation with @cuda.njit)

# 1D Third Order

![1D, Third Order](1d3o.png)

Best to worst for larger arrays:
*  nbconv - numba njit (naive implementation with njit)
*  nbcg - numba cuda (naive implementation with @cuda.njit and grid)
*  nbc - numba cuda (naive implementation with @cuda.njit)

![1D, Third Order](1d3o_g.png)

Best to worst for larger arrays:
*  nbcg - numba cuda (naive implementation with @cuda.njit and grid, 128 is faster than 1024 (t/b))
*  nbconv - numba njit (naive implementation with njit)
