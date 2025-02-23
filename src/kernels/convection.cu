#pragma once
#include "convection.h"
#include "stdio.h"
#include <stdlib.h>
#include <math.h>

#include "deriv.h"

__global__ void check()
{
	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);
	printf("Ra= %f Pr=%f \n", dev.Ra, dev.Pr);
	printf("Rav= %f K=%f Le=%f \n", dev.Rav, dev.K, dev.Le);
	printf("psi1= %f psi2= %f psiS= %f\n", dev.psi1, dev.psi2, dev.psiS);
	printf("Sc1= %f Sc2=%f \n", dev.Sc11, dev.Sc22);
	printf("dim= %i \n", dev.dim);
	printf("hx= %f hy=%f hz=%f \n", dev.hx, dev.hy, dev.hz);
	printf("tau= %20.16f  \n", dev.tau);
	printf("tau_p= %20.16f  \n", dev.tau_p);
	printf("nx= %i ny=%i nz=%i ns=%i N=%i \n", dev.nx, dev.ny, dev.nz, dev.ns, dev.N);
	printf("Lx= %f Ly=%f Lz=%f Lz=%f \n", dev.Lx, dev.Ly, dev.Lz, dev.Ls);
	printf("offset= %i offset2=%i offset3=%i \n", dev.offset, dev.offset2, dev.offset3);
	printf("grav_x= %f grav_y= %f \n", dev.grav_x, dev.grav_y);
	printf("vibr_x= %f vibr_y= %f \n", dev.vibr_x, dev.vibr_y);
	printf("density_x= %f density_y= %f \n", dev.density_x, dev.density_y);
	printf("xbc= %i, ybc= %i, zbc= %i \n", (int)dev.xbc, (int)dev.ybc, (int)dev.zbc);
	printf("A= %f Ca= %f Gr = %f\n", dev.A, dev.Ca, dev.Gr);

	printf("\n");
}

__global__ void swap_one(double* f_old, double* f_new)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < dev.N)
		f_old[l] = f_new[l];
}

__global__ void swap_two(double* f_old, double* f_new, double* f2_old, double* f2_new)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < dev.N)
	{
		f_old[l] = f_new[l];
		f2_old[l] = f2_new[l];
	}	
}

__global__ void swap_three(double* f_old, double* f_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < dev.N)
	{
		f_old[l] = f_new[l];
		f2_old[l] = f2_new[l];
		f3_old[l] = f3_new[l];
	}
}


namespace dim4
{
	__global__ void disturb(unsigned int q, double* f, double value)
	{
		f[q] = value;
	}

	__global__ void check4(unsigned int q, double *T)
	{
		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		printf("check_dim4: %f \n", T[q]);


		if (q < dev.N)
		{

			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				
			}
		}
	}

	__global__ void temperature_check(double* T, double* T0)
	{
		unsigned int q = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		if (q < dev.N)
		{
			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				T[q] = 1;
				
			}
			//boundaries
			else
			{				
				if (j == 0)
				{
					T[q] = 1.0;
					return;
				}
				else if (j == dev.ny)
				{
					T[q] = 0.0;
					return;
				}
				else if (i == 0)
				{
					T[q] = dx1_eq_0_forward(q, T0);
					return;
				}
				else if (i == dev.nx)
				{
					T[q] = dx1_eq_0_back(q, T0);
					return;
				}
				else if (k == 0)
				{
					T[q] = dz1_eq_0_increase(q, T0);
					return;
				}
				else if (k == dev.nz)
				{
					T[q] = dz1_eq_0_decrease(q, T0);
					return;
				}
				else if (l == 0)
				{
					T[q] = ds1_eq_0_increase(q, T0);
					return;
				}
				else if (l == dev.ns)
				{
					T[q] = ds1_eq_0_decrease(q, T0);
					return;
				}
			}

		}
	}

	
	__global__ void temperature(double* T, double* T0, double* vx, double* vy, double* vz, double* vs)
	{
		unsigned int q = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		if (q < dev.N)
		{
			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				T[q] = T0[q]
					+ dev.tau * (
						- VgradF(q, T0, vx, vy, vz, vs)
						+ laplace(q, T0) / dev.Pr
						);
			}
			//boundaries
			else
			{
				if (j == 0)
				{
					T[q] = 1.0;
					return;
				}
				else if (j == dev.ny)
				{
					T[q] = 0.0;
					return;
				}
				else if (i == 0)
				{
					T[q] = dx1_eq_0_forward(q, T0);
					return;
				}
				else if (i == dev.nx)
				{
					T[q] = dx1_eq_0_back(q, T0);
					return;
				}
				else if (k == 0)
				{
					T[q] = dz1_eq_0_increase(q, T0);
					return;
				}
				else if (k == dev.nz)
				{
					T[q] = dz1_eq_0_decrease(q, T0);
					return;
				}
				else if (l == 0)
				{
					T[q] = ds1_eq_0_increase(q, T0);
					return;
				}
				else if (l == dev.ns)
				{
					T[q] = ds1_eq_0_decrease(q, T0);
					return;
				}
			}
			
		}
	}

	__global__ void quasi_velocity(double* T, 
		double* ux, double* uy, double* uz, double *us,
		double* vx, double* vy, double* vz, double *vs) {

		unsigned int q = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		if (q < dev.N)
		{
			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				ux[q] = vx[q] + dev.tau * (-VgradF(q, vx, vx, vy, vz, vs) + laplace(q, vx));
				uy[q] = vy[q] + dev.tau * (-VgradF(q, vy, vx, vy, vz, vs) + laplace(q, vy) + dev.Ra / dev.Pr * (T[q]) * dev.grav_y);
				uz[q] = vz[q] + dev.tau * (-VgradF(q, vz, vx, vy, vz, vs) + laplace(q, vz));
				us[q] = vs[q] + dev.tau * (-VgradF(q, vs, vx, vy, vz, vs) + laplace(q, vs));
				return;
			}
			//boundaries
			else
			{
				if (j == 0)
				{
					uy[q] = dev.tau * dy2_up(q, vy) + dev.tau * dev.Ra / dev.Pr * (T[q]) * dev.grav_y;
					return;
				}
				else if (j == dev.ny)
				{
					uy[q] = dev.tau * dy2_down(q, vy) + dev.tau * dev.Ra / dev.Pr * (T[q]) * dev.grav_y;
					return;
				}
				else if (i == 0)
				{
					ux[q] = dev.tau * dx1_forward(q, vy);
					return;
				}
				else if (i == dev.nx)
				{
					ux[q] = dev.tau * dx1_back(q, vx);
					return;
				}
				else if (k == 0)
				{
					uz[q] = dev.tau * dz1_increase(q, vz);
					return;
				}
				else if (k == dev.nz)
				{
					uz[q] = dev.tau * dz1_decrease(q, vz);
					return;
				}
				else if (l == 0)
				{
					us[q] = dev.tau * ds1_increase(q, vs);
					return;
				}
				else if (l == dev.ns)
				{
					us[q] = dev.tau * ds1_decrease(q, vs);
					return;
				}
			}

		}
	}

	__global__ void velocity_correction(double* p,
		double* ux, double* uy, double* uz, double* us,
		double* vx, double* vy, double* vz, double* vs)
	{
		unsigned int q = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		if (q < dev.N)
		{
			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				vx[q] = ux[q] - dev.tau * dx1(q, p);
				vy[q] = uy[q] - dev.tau * dy1(q, p);
				vz[q] = uz[q] - dev.tau * dz1(q, p);
				vs[q] = us[q] - dev.tau * ds1(q, p);
				return;
			}
			//boundaries
			else
			{
				if (j == 0)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (j == dev.ny)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (i == 0)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (i == dev.nx)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (k == 0)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (k == dev.nz)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (l == 0)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
				else if (l == dev.ns)
				{
					vx[q] = 0.0;
					vy[q] = 0.0;
					vz[q] = 0.0;
					vs[q] = 0.0;
					return;
				}
			}

		}

	}

	__global__ void poisson(double* p, double* p0, double* ux, double* uy, double* uz, double *us)
	{
		unsigned int q = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int q_, i, j, k, l;
		q_ = q;
		l = q_ / dev.offset3;		q_ %= dev.offset3;
		k = q_ / dev.offset2;		q_ %= dev.offset2;
		j = q_ / dev.offset;
		i = q_ % dev.offset;

		double tau_p = 0.05 * dev.hx * dev.hx;

		if (q < dev.N)
		{
			//inner nodes
			if ((i > 0 && i < dev.nx) &&
				(j > 0 && j < dev.ny) &&
				(k > 0 && k < dev.nz) &&
				(l > 0 && l < dev.ns))
			{
				p[q] = -(dx1(q, ux) + dy1(q, uy) + dz1(q, uz) + ds1(q, us)) / dev.tau;

				if (i == 1)					p[q] += 2.0 / 3.0 / dev.hx / dev.hx * (p0[q + 1] - p0[q] - dev.hx / dev.tau * ux[q - 1]);
				else if (i == dev.nx - 1)	p[q] += 2.0 / 3.0 / dev.hx / dev.hx * (p0[q - 1] - p0[q] + dev.hx / dev.tau * ux[q + 1]);
				else						
					p[q] += 1.0 / dev.hx / dev.hx * (p0[q + 1] + p0[q - 1] - 2.0 * p0[q]);

				if (j == 1)					p[q] += 2.0 / 3.0 / dev.hy / dev.hy * (p0[q + dev.offset] - p0[q] - dev.hy / dev.tau * uy[q - dev.offset]);
				else if (j == dev.ny - 1) 	p[q] += 2.0 / 3.0 / dev.hy / dev.hy * (p0[q - dev.offset] - p0[q] + dev.hy / dev.tau * uy[q + dev.offset]);
				else						
					p[q] += 1.0 / dev.hy / dev.hy * (p0[q + dev.offset] + p0[q - dev.offset] - 2.0 * p0[q]);

				if (k == 1)					p[q] += 2.0 / 3.0 / dev.hz / dev.hz * (p0[q + dev.offset2] - p0[q] - dev.hz / dev.tau * uz[q - dev.offset2]);
				else if (k == dev.nz - 1)	p[q] += 2.0 / 3.0 / dev.hz / dev.hz * (p0[q - dev.offset2] - p0[q] + dev.hz / dev.tau * uz[q + dev.offset2]);
				else						
					p[q] += 1.0 / dev.hz / dev.hz * (p0[q + dev.offset2] + p0[q - dev.offset2] - 2.0 * p0[q]);

				if (l == 1)					p[q] += 2.0 / 3.0 / dev.hs / dev.hs * (p0[q + dev.offset3] - p0[q] - dev.hs / dev.tau * us[q - dev.offset3]);
				else if (l == dev.ns - 1)	p[q] += 2.0 / 3.0 / dev.hs / dev.hs * (p0[q - dev.offset3] - p0[q] + dev.hs / dev.tau * us[q + dev.offset3]);
				else						
					p[q] += 1.0 / dev.hs / dev.hs * (p0[q + dev.offset3] + p0[q - dev.offset3] - 2.0 * p0[q]);

				p[q] *= tau_p;
				p[q] += p0[q];

				return;
			}
			//boundaries
			else
			{
				if (j == 0)
				{
					p[q] = dy1_eq_0_up(q, p0) - uy[q] * 2.0 * dev.hy / dev.tau / 3.0;
					return; 
				}
				else if (j == dev.ny)
				{
					p[q] = dy1_eq_0_down(q, p0) + uy[q] * 2.0 * dev.hy / dev.tau / 3.0;
					return;
				}
				else if (i == 0)
				{
					p[q] = dx1_eq_0_forward(q, p0) - ux[q] * 2.0 * dev.hx / dev.tau / 3.0;
					return;
				}
				else if (i == dev.nx)
				{
					p[q] = dx1_eq_0_back(q, p0) + ux[q] * 2.0 * dev.hx / dev.tau / 3.0;
					return;
				}
				else if (k == 0)
				{
					p[q] = dz1_eq_0_increase(q, p0) - uz[q] * 2.0 * dev.hz / dev.tau / 3.0;
					return;
				}
				else if (k == dev.nz)
				{
					p[q] = dz1_eq_0_decrease(q, p0) + uz[q] * 2.0 * dev.hz / dev.tau / 3.0;
					return;
				}
				else if (l == 0)
				{
					p[q] = ds1_eq_0_increase(q, p0) - us[q] * 2.0 * dev.hs / dev.tau / 3.0;
					return;
				}
				else if (l == dev.ns)
				{
					p[q] = ds1_eq_0_decrease(q, p0) + us[q] * 2.0 * dev.hs / dev.tau / 3.0;
					return;
				}
			}

		}

	}

}

