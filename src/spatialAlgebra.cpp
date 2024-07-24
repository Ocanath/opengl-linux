#include "spatialAlgebra.h"
#include "sin_fast.h"
/*
TOOD: Benchmark speed
TODO: implement bottom method using float pointers instead of copying the top and bottom of each into a new location (for speed increase)
*/
vect6_t spatial_vect6_cross(vect6_t a, vect6_t b)
{
	//mat6_t across = spatial_cross_operator(a);
	//vect6_t ret = mat6_vect6_mult(across, b);
	//return ret;
	vect3_t top_a;
	vect3_t bot_a;
	vect3_t top_b;
	vect3_t bot_b;
	int i;
	for (i = 0; i < 3; i++)
	{
		top_a.v[i] = a.v[i];
		bot_a.v[i] = a.v[i + 3];
		top_b.v[i] = b.v[i];
		bot_b.v[i] = b.v[i + 3];
	}
	vect3_t top, res1, res2;
	cross_pbr(&top_a, &top_b, &top);

	cross_pbr(&bot_a, &top_b, &res1);
	cross_pbr(&top_a, &bot_b, &res2);
	vect3_t bot = vect3_add(res1, res2);	

	vect6_t ret;
	for (i = 0; i < 3; i++)
	{
		ret.v[i] = top.v[i];
		ret.v[i + 3] = bot.v[i];
	}
	return ret;
}


mat6_t spatial_cross_operator(vect6_t v)
{
	vect3_t a;
	vect3_t b;
	int i;
	for (i = 0; i < 3; i++)
	{
		a.v[i] = v.v[i];
		b.v[i] = v.v[i + 3];
	}
	mat6_t ret;
	mat3_t ax = skew(a);
	mat3_t bx = skew(b);
	int r, c;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
		{
			ret.m[r][c] = ax.m[r][c];
			ret.m[r + 3][c + 3] = ax.m[r][c];
			ret.m[r + 3][c] = bx.m[r][c];
			ret.m[r][c + 3] = 0;
		}
	}
	return ret;
}

mat6_t h4_to_X(mat4_t ha_b)
{
	mat3_t E;
	int r, c;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
			E.m[r][c] = ha_b.m[r][c];
	}
	vect3_t ob;
	for (r = 0; r < 3; r++)
		ob.v[r] = ha_b.m[r][3];
	mat3_t rx = skew(ob);
	mat3_t rx_E = mat3_mult(rx, E);
	mat6_t ret;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
		{
			ret.m[r][c] = E.m[r][c];
			ret.m[r + 3][c + 3] = E.m[r][c];
			ret.m[r][c + 3] = 0;
		}
	}
	for (r = 3; r < 6; r++)
	{
		for (c = 0; c < 3; c++)
			ret.m[r][c] = rx_E.m[r-3][c];
	}
	return ret;
}
vect6_t c_change_spatial(mat4_t ha_b, vect6_t vb)
{
	vect3_t v, w;
	int r,c;
	for (r = 0; r < 3; r++)
	{
		w.v[r] = vb.v[r];
		v.v[r] = vb.v[r + 3];
	}
	mat3_t E;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
			E.m[r][c] = ha_b.m[r][c];
	}
	vect3_t w_ret = mat3_vect3_mult(E, w);
	vect3_t ob;
	for (r = 0; r < 3; r++)
		ob.v[r] = ha_b.m[r][3];
	mat3_t rx = skew(ob);
	vect3_t v_ret = vect3_add(mat3_vect3_mult(rx, mat3_vect3_mult(E, w)), mat3_vect3_mult(E,v));
	vect6_t ret;
	for (r = 0; r < 3; r++)
	{
		ret.v[r] = w_ret.v[r];
		ret.v[r + 3] = v_ret.v[r];
	}
	return ret;
}

mat3_t mat3_T(mat3_t in)
{
	mat3_t ret;
	int r, c;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
		{
			ret.m[r][c] = in.m[c][r];
		}
	}
	return ret;
}


mat3_t Rx(float angle)
{
	mat3_t ret;
	ret.m[0][0] = 1;	ret.m[0][1] = 0;			ret.m[0][2] = 0;			
	ret.m[1][0] = 0;	ret.m[1][1] = cos_fast(angle);	ret.m[1][2] = -sin_fast(angle);
	ret.m[2][0] = 0;	ret.m[2][1] = sin_fast(angle);	ret.m[2][2] = cos_fast(angle);
	return ret;
}
/*Returns rotation about coordinate. 0 = identity*/
mat3_t Ry(float angle)
{
	mat3_t ret;
	ret.m[0][0] = cos_fast(angle);	ret.m[0][1] = 0;	ret.m[0][2] = sin_fast(angle);
	ret.m[1][0] = 0;			ret.m[1][1] = 1;	ret.m[1][2] = 0;			
	ret.m[2][0] = -sin_fast(angle);	ret.m[2][1] = 0;	ret.m[2][2] = cos_fast(angle);
	return ret;
}
/*Returns rotation about coordinate. 0 = identity*/
mat3_t Rz(float angle)
{
	mat3_t ret;
	ret.m[0][0] = cos_fast(angle);	ret.m[0][1] = -sin_fast(angle);		ret.m[0][2] = 0;
	ret.m[1][0] = sin_fast(angle);	ret.m[1][1] = cos_fast(angle);		ret.m[1][2] = 0;
	ret.m[2][0] = 0;			ret.m[2][1] = 0;				ret.m[2][2] = 1;	

	return ret;
}

/*returns the inverse of a homogeneous transform type mat4_t matrix*/
mat4_t ht_inverse(mat4_t hin)
{
	mat4_t hout;
	int r; int c;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
		{
			hout.m[r][c] = hin.m[c][r];
		}
	}
	hout.m[0][3] = -(hout.m[0][0] * hin.m[0][3] + hout.m[0][1] * hin.m[1][3] + hout.m[0][2] * hin.m[2][3]);
	hout.m[1][3] = -(hout.m[1][0] * hin.m[0][3] + hout.m[1][1] * hin.m[1][3] + hout.m[1][2] * hin.m[2][3]);
	hout.m[2][3] = -(hout.m[2][0] * hin.m[0][3] + hout.m[2][1] * hin.m[1][3] + hout.m[2][2] * hin.m[2][3]);

	hout.m[3][0] = 0; hout.m[3][1] = 0; hout.m[3][2] = 0; hout.m[3][3] = 1.0;
	return hout;
}

/*returns the inverse of a homogeneous transform type mat4_t matrix*/
void ht_inverse_ptr(mat4_t * hin, mat4_t * hout)
{
	int r; int c;
	for (r = 0; r < 3; r++)
	{
		for (c = 0; c < 3; c++)
		{
			hout->m[r][c] = hin->m[c][r];
		}
	}
	hout->m[0][3] = -(hout->m[0][0] * hin->m[0][3] + hout->m[0][1] * hin->m[1][3] + hout->m[0][2] * hin->m[2][3]);
	hout->m[1][3] = -(hout->m[1][0] * hin->m[0][3] + hout->m[1][1] * hin->m[1][3] + hout->m[1][2] * hin->m[2][3]);
	hout->m[2][3] = -(hout->m[2][0] * hin->m[0][3] + hout->m[2][1] * hin->m[1][3] + hout->m[2][2] * hin->m[2][3]);

	hout->m[3][0] = 0; hout->m[3][1] = 0; hout->m[3][2] = 0; hout->m[3][3] = 1.0;
}

/*returns homogeneous transform mat4_t matrix which is the rotation 'analge' around the x axis */
mat4_t Hx(float angle)
{
	mat4_t ret;
	ret.m[0][0] = 1;	ret.m[0][1] = 0;				ret.m[0][2] = 0;				ret.m[0][3] = 0;
	ret.m[1][0] = 0;	ret.m[1][1] = cos_fast(angle);	ret.m[1][2] = -sin_fast(angle);	ret.m[1][3] = 0;
	ret.m[2][0] = 0;	ret.m[2][1] = sin_fast(angle);	ret.m[2][2] = cos_fast(angle);	ret.m[2][3] = 0;
	ret.m[3][0] = 0;	ret.m[3][1] = 0;				ret.m[3][2] = 0;				ret.m[3][3] = 1;
	return ret;
}

/*Returns rotation about coordinate. 0 = identity*/
mat4_t Hy(float angle)
{
	mat4_t ret;
	ret.m[0][0] = cos_fast(angle);	ret.m[0][1] = 0;	ret.m[0][2] = sin_fast(angle);	ret.m[0][3] = 0;
	ret.m[1][0] = 0;				ret.m[1][1] = 1;	ret.m[1][2] = 0;				ret.m[1][3] = 0;
	ret.m[2][0] = -sin_fast(angle);	ret.m[2][1] = 0;	ret.m[2][2] = cos_fast(angle);	ret.m[2][3] = 0;
	ret.m[3][0] = 0;				ret.m[3][1] = 0;	ret.m[3][2] = 0;				ret.m[3][3] = 1;
	return ret;	
}
/*Returns rotation about coordinate. 0 = identity*/
mat4_t Hz(float angle)
{
	mat4_t ret;
	ret.m[0][0] = cos_fast(angle);		ret.m[0][1] = -sin_fast(angle);		ret.m[0][2] = 0;	ret.m[0][3] = 0;
	ret.m[1][0] = sin_fast(angle);		ret.m[1][1] = cos_fast(angle);		ret.m[1][2] = 0;	ret.m[1][3] = 0;
	ret.m[2][0] = 0;					ret.m[2][1] = 0;					ret.m[2][2] = 1;	ret.m[2][3] = 0;
	ret.m[3][0] = 0;					ret.m[3][1] = 0;					ret.m[3][2] = 0;	ret.m[3][3] = 1;
	return ret;
}

/*
* returns a scale mat4 matrixxx
*/
mat4_t Hscale(float scf)
{
	mat4_t H_scf = {
		{
			{scf, 0, 0, 0},
			{0, scf, 0, 0},
			{0, 0, scf, 0},
			{0, 0, 0, 1.f}
		}
	};
	return H_scf;
}

/*
	Returns vector cross product between 3 vectors A and B.
*/
vect3_t cross(vect3_t v_a, vect3_t v_b)
{
	vect3_t ret;
	ret.v[0] = -v_a.v[2] * v_b.v[1] + v_a.v[1] * v_b.v[2];
	ret.v[1] = v_a.v[2] * v_b.v[0] - v_a.v[0] * v_b.v[2];
	ret.v[2] = -v_a.v[1] * v_b.v[0] + v_a.v[0] * v_b.v[1];
	return ret;
}



/*
	Returns vector cross product between 3 vectors A and B. Faster pass by pointer version
*/
void cross_pbr(vect3_t * v_a, vect3_t * v_b, vect3_t * ret)
{
	ret->v[0] = -v_a->v[2]*v_b->v[1] + v_a->v[1]*v_b->v[2];
	ret->v[1] = v_a->v[2]*v_b->v[0] - v_a->v[0]*v_b->v[2];
	ret->v[2] = -v_a->v[1]*v_b->v[0] + v_a->v[0]*v_b->v[1];
}
/*Returns skew symmetric matrix of 3 vector vin*/
mat3_t skew(vect3_t vin)
{
	mat3_t ret;
	float x = vin.v[0];	float y = vin.v[1];	float z = vin.v[2];
	ret.m[0][0] = 0;	ret.m[0][1] = -z;	ret.m[0][2] = y;
	ret.m[1][0] = z;	ret.m[1][1] = 0;	ret.m[1][2] = -x;
	ret.m[2][0] = -y;	ret.m[2][1] = x;	ret.m[2][2] = 0;
	return ret;
}
/*Multiples vector v_a by scalar scale*/
vect6_t vect6_scale(vect6_t v_a, float scale)
{
	vect6_t ret; int i;
	for (i = 0; i<6; i++)
		ret.v[i] = v_a.v[i] * scale;
	return ret;
}
/*Multiples vector v_a by scalar scale*/
vect4_t vect4_scale(vect4_t v_a, float scale)
{
	vect4_t ret; int i;
	for(i=0;i<4;i++)
		ret.v[i] = v_a.v[i] * scale;
	return ret;
}
/*Multiples vector v_a by scalar scale*/
vect3_t vect3_scale(vect3_t v_a, float scale)
{
	vect3_t ret; int i;
	for (i = 0; i<3; i++)
		ret.v[i] = v_a.v[i] * scale;
	return ret;
}

vect6_t vect6_add(vect6_t v_a, vect6_t v_b)
{
	vect6_t ret;
	int dim = 6; int i;
	for (i = 0; i < dim; i++)
		ret.v[i] = v_a.v[i] + v_b.v[i];
	return ret;
}

vect4_t vect4_add(vect4_t v_a, vect4_t v_b)
{
	vect4_t ret;
	int dim = 4; int i;
	for (i = 0; i < dim; i++)
		ret.v[i] = v_a.v[i] + v_b.v[i];
	return ret;
}

vect6_t zero_vect6()
{
	vect6_t ret;
	int i;
	for (i = 0; i < 6; i++)
		ret.v[i] = 0;
	return ret;
}

vect3_t vect3_add(vect3_t v_a, vect3_t v_b)
{
	vect3_t ret;
	int dim = 3; int i;
	for (i = 0; i < dim; i++)
		ret.v[i] = v_a.v[i] + v_b.v[i];
	return ret;
}


/*Dot product. floating point*/
float vect_dot(float* v1, float* v2, int n)
{
	float res = 0.f;
	for (int i = 0; i < n; i++)
	{
		res += v1[i] * v2[i];
	}
	return res;
}

/**/
float vect_mag(float* v, int n)
{
	float v_dot_v= 0.f;
	for (int i = 0; i < n; i++)
		v_dot_v += v[i] * v[i];
	return (float)sqrt((double)v_dot_v);
}


typedef union u32_fmt_t
{
	uint32_t u32;
	int32_t i32;
	float f32;
	int16_t i16[sizeof(uint32_t) / sizeof(int16_t)];
	uint16_t ui16[sizeof(uint32_t) / sizeof(uint16_t)];
	int8_t i8[sizeof(uint32_t) / sizeof(int8_t)];
	uint8_t ui8[sizeof(uint32_t) / sizeof(uint8_t)];
}u32_fmt_t;

float Q_rsqrt(float number)
{
	u32_fmt_t conv;
	conv.f32 = number;
	conv.u32 = 0x5f3759df - (conv.u32 >> 1);
	conv.f32 *= 1.5F - (number * 0.5F * conv.f32 * conv.f32);
	return conv.f32;
}


/**/
float inverse_vect_mag(float* v, int n)
{
	float v_dot_v = 0.f;
	for (int i = 0; i < n; i++)
		v_dot_v += v[i] * v[i];
	return Q_rsqrt(v_dot_v);
}


/**/
void vect_normalize(float* v, int n)
{
	float inv_mag = inverse_vect_mag(v, n);
	for (int i = 0; i < n; i++)
	{
		v[i] = v[i] * inv_mag;
	}
}


float vect3_magnitude(vect3_t v)
{
	return sqrt(v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2]);
}
vect3_t vect3_normalize(vect3_t v)
{
	float scale = 1 / (sqrt(v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2]));
	v.v[0] *= scale;
	v.v[1] *= scale;
	v.v[2] *= scale;
	return v;
}

vect6_t mat6_vect6_mult(mat6_t m, vect6_t v)
{
	int rsize = 6; int vsize = 6;
	vect6_t ret;
	int r, i;
	for (r = 0; r < rsize; r++)
	{
		float tmp = 0;
		for (i = 0; i < vsize; i++)
		{
			tmp = tmp + m.m[r][i] * v.v[i];
		}
		ret.v[r] = tmp;
	}
	return ret;
}

vect4_t mat4_t_vect4_mult(mat4_t m, vect4_t v)
{
	int rsize = 4; int vsize = 4;
	vect4_t ret;
	int r, i;
	for (r = 0; r < rsize; r++)
	{
		float tmp = 0;
		for (i = 0; i < vsize; i++)
		{
			tmp = tmp + m.m[r][i] * v.v[i];
		}
		ret.v[r] = tmp;
	}
	return ret;
}
vect3_t mat4_t_vect3_mult(mat4_t m, vect3_t v)
{
	int rsize = 3; int vsize = 4;
	vect3_t ret;
	int r, i;
	for (r = 0; r < rsize; r++)
	{
		float tmp = 0;
		for (i = 0; i < vsize; i++)
		{
			if (i < 3)
				tmp = tmp + m.m[r][i] * v.v[i];
			if (i == 3)
				tmp = tmp + m.m[r][i];
		}
		ret.v[r] = tmp;
	}
	return ret;
	return ret;
}

vect3_t mat3_vect3_mult(mat3_t m, vect3_t v)
{
	int rsize = 3; int vsize = 3;
	vect3_t ret;
	int r, i;
	for (r = 0; r < rsize; r++)
	{
		float tmp = 0;
		for (i = 0; i < vsize; i++)
		{
			tmp = tmp + m.m[r][i] * v.v[i];
		}
		ret.v[r] = tmp;
	}
	return ret;
}

mat6_t mat6_mult(mat6_t m1, mat6_t m2)
{
	mat6_t ret;
	int dim = 6;
	int out_r; int out_c; int i;
	for (out_r = 0; out_r < dim; out_r++)
	{
		for (out_c = 0; out_c < dim; out_c++)
		{
			float tmp = 0;
			for (i = 0; i < dim; i++)
			{
				tmp = tmp + m1.m[out_r][i] * m2.m[i][out_c];
			}
			ret.m[out_r][out_c] = tmp;
		}
	}
	return ret;
}
/*
	Multiplies two mat4_t matrices
*/
mat4_t mat4_t_mult(mat4_t m1, mat4_t m2)
{
	mat4_t ret;
	int dim = 4;
	int out_r; int out_c; int i;
	for (out_r = 0; out_r < dim; out_r++)
	{
		for (out_c = 0; out_c < dim; out_c++)
		{
			float tmp = 0;
			for (i = 0; i < dim; i++)
			{
				tmp = tmp + m1.m[out_r][i] * m2.m[i][out_c];
			}
			ret.m[out_r][out_c] = tmp;
		}
	}
	return ret;
}
/*
	Multiplies two mat4_t matrices, pass by pointer
*/
void mat4_t_mult_pbr(mat4_t * m1, mat4_t * m2, mat4_t * ret)
{
	int dim = 4;
	int out_r; int out_c; int i;
	for (out_r = 0; out_r < dim; out_r++)
	{
		for (out_c = 0; out_c < dim; out_c++)
		{
			float tmp = 0;
			for (i = 0; i < dim; i++)
			{
				tmp = tmp + m1->m[out_r][i] * m2->m[i][out_c];
			}
			ret->m[out_r][out_c] = tmp;
		}
	}
}

mat3_t mat3_mult(mat3_t m1, mat3_t m2)
{
	mat3_t ret;
	int dim = 3;
	int out_r; int out_c; int i;
	for (out_r = 0; out_r < dim; out_r++)
	{
		for (out_c = 0; out_c < dim; out_c++)
		{
			float tmp = 0;
			for (i = 0; i < dim; i++)
			{
				tmp = tmp + m1.m[out_r][i] * m2.m[i][out_c];
			}
			ret.m[out_r][out_c] = tmp;
		}
	}
	return ret;
}

mat6_t mat6_add(mat6_t m1, mat6_t m2)
{
	mat6_t ret;
	int dim = 6;
	int r, c;
	for (r = 0; r < dim; r++)
	{
		for (c = 0; c < dim; c++)
		{
			ret.m[r][c] = m1.m[r][c] + m2.m[r][c];
		}
	}
	return ret;
}

mat4_t mat4_t_add(mat4_t m1, mat4_t m2)
{
	mat4_t ret;
	int dim = 4;
	int r, c;
	for (r = 0; r < dim; r++)
	{
		for (c = 0; c < dim; c++)
		{
			ret.m[r][c] = m1.m[r][c] + m2.m[r][c];
		}
	}
	return ret;
}

mat3_t mat3_add(mat3_t m1, mat3_t m2)
{
	mat3_t ret;
	int dim = 3;
	int r, c;
	for (r = 0; r < dim; r++)
	{
		for (c = 0; c < dim; c++)
		{
			ret.m[r][c] = m1.m[r][c] + m2.m[r][c];
		}
	}
	return ret;
}



const mat4_t mat4_t_Identity = {
	{
		{ 1, 0, 0, 0 },
		{ 0, 1, 0, 0 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 1 }
	}
};
const mat3_t mat3_Identity = {
	{
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 }
	}
};
mat4_t mat4_t_I()
{
	return mat4_t_Identity;
}
mat3_t mat3_I()
{
	return mat3_Identity;
}


void matrix_vect_multiply(float ** M1, int rsize, float * v, int vsize, float * out)
{
	//multiply
	int r, i;
	for (r = 1; r <= rsize; r++)
	{
		float tmp = 0;
		for (i = 1; i <= vsize; i++)
		{
			tmp = tmp + M1[r - 1][i - 1] * v[i];
		}
		out[r] = tmp;
	}
}


vect3_t vect4_to_vect3(vect4_t in)
{
	vect3_t ret;
	int i;
	for (i = 0; i < 3; i++)
		ret.v[i] = in.v[i];
	return ret;
}

vect4_t vect3_to_vect4(vect3_t in)
{
	vect4_t ret;
	int i;
	for (i = 0; i < 3; i++)
		ret.v[i] = in.v[i];
	ret.v[3] = 1;
	return ret;
}