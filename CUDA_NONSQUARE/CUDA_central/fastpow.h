//
//   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
//
//   CUDA code for massively parallelized Monte Carlo simulation of
//   two-dimensional disks
//
//   This is a C++ template that generates powers of argument for integer
//   arguments, which must be known at compile time. We edited the code from
//   http://szhorvat.net/pelican/fast-computation-of-powers.html
//
//   URL: https://github.com/jellyfysh/SoftDisks
//   See LICENSE for copyright information
//
//   If you use this code or find it useful, please cite the following paper:
//
//   @article{PhysRevE.108.024103,
//       title = {Liquid-hexatic transition for soft disks},
//       author = {Nishikawa, Yoshihiko and Krauth, Werner and Maggs, A. C.},
//       journal = {Phys. Rev. E},
//       volume = {108},
//       issue = {2},
//       pages = {024103},
//       numpages = {7},
//       year = {2023},
//       month = {Aug},
//       publisher = {American Physical Society},
//       doi = {10.1103/PhysRevE.108.024103},
//       url = {https://link.aps.org/doi/10.1103/PhysRevE.108.024103}
//   }
//
//

///////// Power function for energy calculation /////////
template <unsigned N> struct e_power_impl;

template <unsigned N> struct e_power_impl {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    if (N % 2 == 0)
      return e_power_impl<N / 2>::calc(x * x);
    else if (N % 3 == 0)
      return e_power_impl<N / 3>::calc(x * x * x);
    return e_power_impl<N - 1>::calc(x) * x;
  }
};

template <> struct e_power_impl<0> {
  template <typename T> static __host__ __device__ T calc(const T &) {
    return 1;
  }
};

template <> struct e_power_impl<612 / 2> {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    T x6 = x * x * x;
    T x12 = x6 * x6;
    return 4. * (x12 - x6);
  }
};

template <unsigned N, typename T>
__host__ __device__ inline T e_power(const T &x) {
  return e_power_impl<N>::calc(x);
}

// powq is used in the CUDA code, here I define it as a inline which calls down
// to template solution given above, this only works for even powers, no sqrt

template <typename T>
__host__ __device__ inline T powq(T nexdist, float) { // specific code for n=12
  T r2 = 1. / nexdist;
  return e_power<rpotential / 2>(r2); // rpotential comes from the Script/Run.sh
}

// this is a solution for odd powers, note rsqrt is much faster than sqrt.
template <typename T>
__host__ __device__ inline T poww(T nexdist, float) { // specific code for n=12
  T r2 = 1. / nexdist;
  return rsqrt(nexdist) * e_power<(rpotential - 1) / 2>(r2);
}

///////// Power function for pressure calculation /////////
template <unsigned N> struct p_power_impl;

template <unsigned N> struct p_power_impl {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    if (N % 2 == 0)
      return p_power_impl<N / 2>::calc(x * x);
    else if (N % 3 == 0)
      return p_power_impl<N / 3>::calc(x * x * x);
    return p_power_impl<N - 1>::calc(x) * x;
  }
};

template <> struct p_power_impl<0> {
  template <typename T> static __host__ __device__ T calc(const T &) {
    return rpotential;
  }
};

template <> struct p_power_impl<612 / 2> {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    T x6 = x * x * x;
    T x12 = x6 * x6;
    return 4. * (12. * x12 - 6. * x6);
  }
};

template <unsigned N, typename T>
__host__ __device__ inline T p_power(const T &x) {
  return p_power_impl<N>::calc(x);
}

// powp is used in the pressure code

template <typename T>
__host__ __device__ inline T powp(T nexdist, float) { // specific code for n=12
  T r2 = 1. / nexdist;
  return p_power<rpotential / 2>(r2); // rpotential comes from the Script/Run.sh
}

///////// Power function for hypervirial calculation /////////
template <unsigned N> struct h_power_impl;

template <unsigned N> struct h_power_impl {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    if (N % 2 == 0)
      return h_power_impl<N / 2>::calc(x * x);
    else if (N % 3 == 0)
      return h_power_impl<N / 3>::calc(x * x * x);
    return h_power_impl<N - 1>::calc(x) * x;
  }
};

template <> struct h_power_impl<0> {
  template <typename T> static __host__ __device__ T calc(const T &) {
    return rpotential * (rpotential + 1);
  }
};

template <> struct h_power_impl<612 / 2> {
  template <typename T> static __host__ __device__ T calc(const T &x) {
    T x6 = x * x * x;
    T x12 = x6 * x6;
    return 4. * (156. * x12 - 42. * x6);
  }
};

template <unsigned N, typename T>
__host__ __device__ inline T h_power(const T &x) {
  return h_power_impl<N>::calc(x);
}

// powp is used in the pressure code

template <typename T>
__host__ __device__ inline T powh(T nexdist, float) { // specific code for n=12
  T r2 = 1. / nexdist;
  return h_power<rpotential / 2>(r2); // rpotential comes from the Script/Run.sh
}
