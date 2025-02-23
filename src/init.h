#pragma once



void init_parameters(Configuration& c)
{
	#ifdef __linux__ 
	string str = "mkdir -p fields/";
	system(str.c_str());
	#endif

	#ifdef _WIN32
	CreateDirectoryA("fields", NULL);
	#endif


	ReadingFile par("parameters.txt");

	int bc;
	par.reading<int>(bc, "xbc", 0); c.xbc = static_cast<bc_type>(bc);
	par.reading<int>(bc, "ybc", 0); c.ybc = static_cast<bc_type>(bc);
	par.reading<int>(bc, "zbc", 0); c.zbc = static_cast<bc_type>(bc);
	par.reading<int>(bc, "sbc", 0); c.sbc = static_cast<bc_type>(bc);

	par.reading<double>(c.Lx, "Lx", 1.0);
	par.reading<double>(c.Ly, "Ly", 1.0);
	par.reading<double>(c.Lz, "Lz", 1.0);
	par.reading<double>(c.Ls, "Ls", 1.0);

	par.reading<unsigned int>(c.nx, "nx", 20);
	par.reading<unsigned int>(c.ny, "ny", 20);
	par.reading<unsigned int>(c.nz, "nz", 0);
	par.reading<unsigned int>(c.ns, "ns", 0);

	par.reading<double>(c.Ra, "Ra", 5000);
	par.reading<double>(c.Pr, "Pr", 10);
	par.reading<double>(c.tau, "tau", 0.0001);

	par.reading<double>(c.grav_y, "grav_y", 1.0);

	if (c.nx > 0) c.dim = 1;
	if (c.ny > 0) c.dim = 2;
	if (c.nz > 0) c.dim = 3;
	if (c.ns > 0) c.dim = 4;

	c.hx = c.nx == 0 ? 0 : c.Lx / c.nx;
	c.hy = c.ny == 0 ? 0 : c.Ly / c.ny;
	c.hz = c.nz == 0 ? 0 : c.Lz / c.nz;
	c.hs = c.ns == 0 ? 0 : c.Ls / c.ns;



	if (c.dim >= 2)
	{
		c.Sx = c.hy;
		c.Sy = c.hx;
		c.Sz = 0;
		c.Ss = 0;

		c.dV = c.hx * c.hy;
		c.N = (c.nx + 1) * (c.ny + 1);
		c.offset = c.nx + 1;
		c.offset2 = c.offset3 = 0;
	}

	if (c.dim >= 3)
	{
		c.Sx *= c.hz;
		c.Sy *= c.hz;
		c.Sz = c.hx * c.hy;

		c.dV *= c.hz;
		c.N *= (c.nz + 1);
		c.offset = c.nx + 1;
		c.offset2 = (c.nx + 1) * (c.ny + 1);
	}

	if (c.dim >= 4)
	{
		c.Sx *= c.hs;
		c.Sy *= c.hs;
		c.Sz *= c.hs;
		c.Ss = c.hx * c.hy * c.hz;
		c.dV *= c.hs;
		c.N *= (c.ns + 1);
		c.offset3 = (c.nx + 1) * (c.ny + 1) * (c.nz + 1);
	}

	c.Nbytes = c.N * sizeof(double);

	double pi = 3.1415926535897932384626433832795;
	auto make_vector = [&pi](double angle, double (*func)(double))
	{	return std::floor(func(angle * pi / 180.0) * 1e+10) / 1e+10;	};
	auto set_angles = [&make_vector, &c](double a, double b)
	{
		c.grav_y = make_vector(a, cos);
		c.grav_x = make_vector(a, sin);

		c.vibr_y = make_vector(b, cos);
		c.vibr_x = make_vector(b, sin);

		c.density_y = make_vector(a, cos);
		c.density_x = make_vector(a, sin);
	};

	set_angles(c.alpha, c.beta);





}

void init_fields(Configuration& c, Arrays& h, Arrays& d)
{
	for (unsigned int l = 0; l <= c.ns; l++) {
		for (unsigned int k = 0; k <= c.nz; k++) {
			for (unsigned int j = 0; j <= c.ny; j++) {
				for (unsigned int i = 0; i <= c.nx; i++)
				{
					unsigned int q = INDEX(i, j, k, l);
					double y = c.hx * j;
					h.T[q] = 1.0 - y;
				}
			}
		}
	}
	copyArrayFromHostToDevice({ d.T, d.T0 }, h.T, c.N);
}