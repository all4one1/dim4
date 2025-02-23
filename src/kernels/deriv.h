#pragma once


__device__  double dx1(unsigned int l, double* f) {
	return 0.5 * (f[l + 1] - f[l - 1]) / dev.hx;
}
__device__  double dy1(unsigned int l, double* f) {
	return 0.5 * (f[l + dev.offset] - f[l - dev.offset]) / dev.hy;
}
__device__  double dz1(unsigned int l, double* f) {
	return 0.5 * (f[l + dev.offset2] - f[l - dev.offset2]) / dev.hz;
}
__device__  double ds1(unsigned int l, double* f) {
	return 0.5 * (f[l + dev.offset3] - f[l - dev.offset3]) / dev.hs;
}


__device__  double dx1_forward(unsigned int l, double* f) {
	return -0.5 * (3.0 * f[l] - 4.0 * f[l + 1] + f[l + 2]) / dev.hx;
}
__device__  double dx1_back(unsigned int l, double* f) {
	return  0.5 * (3.0 * f[l] - 4.0 * f[l - 1] + f[l - 2]) / dev.hx;
}
__device__  double dy1_up(unsigned int l, double* f) {
	return  -0.5 * (3.0 * f[l] - 4.0 * f[l + dev.offset] + f[l + dev.offset * 2]) / dev.hy;
}
__device__  double dy1_down(unsigned int l, double* f) {
	return  0.5 * (3.0 * f[l] - 4.0 * f[l - dev.offset] + f[l - dev.offset * 2]) / dev.hy;
}
__device__  double dz1_increase(unsigned int l, double* f) {
	return  -0.5 * (3.0 * f[l] - 4.0 * f[l + dev.offset2] + f[l + dev.offset2 * 2]) / dev.hz;
}
__device__  double dz1_decrease(unsigned int l, double* f) {
	return  0.5 * (3.0 * f[l] - 4.0 * f[l - dev.offset2] + f[l - dev.offset2 * 2]) / dev.hz;
}
__device__  double ds1_increase(unsigned int l, double* f) {
	return  -0.5 * (3.0 * f[l] - 4.0 * f[l + dev.offset3] + f[l + dev.offset3 * 2]) / dev.hs;
}
__device__  double ds1_decrease(unsigned int l, double* f) {
	return  0.5 * (3.0 * f[l] - 4.0 * f[l - dev.offset3] + f[l - dev.offset3 * 2]) / dev.hs;
}


__device__  double dx1_eq_0_forward(unsigned int l, double* f) {
	return (4.0 * f[l + 1] - f[l + 2]) / 3.0;
}
__device__  double dx1_eq_0_back(unsigned int l, double* f) {
	return (4.0 * f[l - 1] - f[l - 2]) / 3.0;
}

__device__  double dy1_eq_0_up(unsigned int l, double* f) {
	return (4.0 * f[l + dev.offset] - f[l + 2 * dev.offset]) / 3.0;
}
__device__  double dy1_eq_0_down(unsigned int l, double* f) {
	return (4.0 * f[l - dev.offset] - f[l - 2 * dev.offset]) / 3.0;
}

__device__  double dz1_eq_0_increase(unsigned int l, double* f) {
	return (4.0 * f[l + dev.offset2] - f[l + 2 * dev.offset2]) / 3.0;
}
__device__  double dz1_eq_0_decrease(unsigned int l, double* f) {
	return (4.0 * f[l - dev.offset2] - f[l - 2 * dev.offset2]) / 3.0;
}
__device__  double ds1_eq_0_increase(unsigned int l, double* f) {
	return (4.0 * f[l + dev.offset3] - f[l + 2 * dev.offset3]) / 3.0;
}
__device__  double ds1_eq_0_decrease(unsigned int l, double* f) {
	return (4.0 * f[l - dev.offset3] - f[l - 2 * dev.offset3]) / 3.0;
}



__device__  double dx2(unsigned int l, double* f) {
	return (f[l + 1] - 2.0 * f[l] + f[l - 1]) / (dev.hx * dev.hx);
}
__device__  double dy2(unsigned int l, double* f) {
	return (f[l + dev.offset] - 2.0 * f[l] + f[l - dev.offset]) / (dev.hy * dev.hy);
}
__device__  double dz2(unsigned int l, double* f) {
	return (f[l + dev.offset2] - 2.0 * f[l] + f[l - dev.offset2]) / (dev.hz * dev.hz);
}
__device__  double ds2(unsigned int l, double* f) {
	return (f[l + dev.offset3] - 2.0 * f[l] + f[l - dev.offset3]) / (dev.hs * dev.hs);
}

__device__  double dx2_forward(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l + 1] + 4.0 * f[l + 2] - f[l + 3]) / (dev.hx * dev.hx);
}
__device__  double dx2_back(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l - 1] + 4.0 * f[l - 2] - f[l - 3]) / (dev.hx * dev.hx);
}
__device__  double dy2_up(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l + dev.offset] + 4.0 * f[l + 2 * dev.offset] - f[l + 3 * dev.offset]) / (dev.hy * dev.hy);
}
__device__  double dy2_down(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l - dev.offset] + 4.0 * f[l - 2 * dev.offset] - f[l - 3 * dev.offset]) / (dev.hy * dev.hy);
}
__device__  double dz2_increase(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l + dev.offset2] + 4.0 * f[l + 2 * dev.offset2] - f[l + 3 * dev.offset2]) / (dev.hz * dev.hz);
}
__device__  double dz2_decrease(unsigned int l, double* f) {
	return (2.0 * f[l] - 5.0 * f[l - dev.offset2] + 4.0 * f[l - 2 * dev.offset2] - f[l - 3 * dev.offset2]) / (dev.hz * dev.hz);
}



__device__ double d2xy(unsigned int l, double* f)
{
	return (f[l + 1 + dev.offset] + f[l - 1 - dev.offset] - f[l + 1 - dev.offset] - f[l - 1 + dev.offset]) / (4.0 * dev.hx * dev.hy);
}

__device__ double d2xz(unsigned int l, double* f)
{
	return (f[l + 1 + dev.offset2] + f[l - 1 - dev.offset2] - f[l + 1 - dev.offset2] - f[l - 1 + dev.offset2]) / (4.0 * dev.hx * dev.hz);
}

__device__ double d2yz(unsigned int l, double* f)
{
	return (f[l + dev.offset + dev.offset2] + f[l - dev.offset - dev.offset2] - f[l + dev.offset - dev.offset2] - f[l - dev.offset + dev.offset2]) / (4.0 * dev.hy * dev.hz);
}


__device__ double VgradF(unsigned int l, double* f, double* vx, double* vy)
{
	return (vx[l] * dx1(l, f) + vy[l] * dy1(l, f));
}

__device__ double VgradF(unsigned int l, double* f, double* vx, double* vy, double* vz)
{
	return (vx[l] * dx1(l, f) + vy[l] * dy1(l, f) + vz[l] * dz1(l, f));
}

__device__ double VgradF(unsigned int l, double* f, double* vx, double* vy, double* vz, double *vs)
{
	return (vx[l] * dx1(l, f) + vy[l] * dy1(l, f) + vz[l] * dz1(l, f) + vs[l] * ds1(l, f));
}

__device__ double laplace(unsigned int l, double* f)
{
	return (dx2(l, f) + dy2(l, f) + dz2(l, f) + ds2(l, f));
}