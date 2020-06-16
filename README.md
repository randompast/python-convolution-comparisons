# Python Convolution Comparisons

Convolutions are a fundamental operation in scientific computing.  The goal here is to explore the possible approaches in python.  A convolution is a sliding dot product.


# 1D First Order
An example 1D signal could be audio amplitudes.  An example filter would be an impulse response.  The filter slides over a window of the larger audio signal.  Corresponding terms are multiplied and then summed together to construct a filtered version of the signal.

It was found that numpy.convolve is the fastest for a large variety of inputs, but after a sufficiently large size it makes sense to use cupy.  The gpu implementations started to become worth using at around 10^7.

![1D, first order](1d1o.png)

Best to worst for larger arrays:
*  cpste - cupy strided
*  nbc_gm, nbc - numba cuda improvement from [@gmarkall](https://github.com/gmarkall)
*  cpv, spfcv, spvc, npc
  *  cpv - cupyx.scipy.ndimage,
  *  spvc - scipy.signal.convolve
  *  spfvc - scipy.signal.fftconvolve
  *  npc - numpy.convolve
*  npste - numpy strided
*  nbpd_np - numba parallel dot, no gil
*  nbp - numba parallel
*  nbnpc_nj - @njit numba numpy convolve
*  npd - numpy dot
*  npfe - numpy for einsum
*  naive - (not shown) two for loops, significantly worse

# 1D Second Order


# 1D Third Order
