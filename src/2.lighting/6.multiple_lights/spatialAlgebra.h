#ifndef SPATIALALGEBRA_H
#define SPATIALALGEBRA_H

#include <math.h>

#define X_IDX 0
#define Y_IDX 0
#define Z_IDX 0

#define PI 3.14159265359f

/*
	Rotation matrices, skew symmetric matrices take this type
*/
typedef float mat3[3][3];
typedef struct mat3_t
{
	mat3 m;
}mat3_t;


/*
	Array wrapper for unambiguous pointer to 2d array
*/
typedef float mat4[4][4];

/*
	homogeneous transformation matrices take this type. Allows working with arrays as a type directly (including returns) instead of pointers
*/
typedef struct mat4_t
{
	mat4 m;
}mat4_t;


/*
	spatial coordinate transformations and spatial inertia tensors take this type.
	NOTE: future implementations shouldn not use this variable type, instead opting for optimizations that are mathematically equivalent
	(such as storage of a spatial transformation as a 1 mat3_t, and a 3 vector)
*/
typedef float mat6[6][6];
typedef struct mat6_t
{
	mat6 m;
}mat6_t;

/*
	Line, free vectors. 
*/
typedef float vect3[3];
typedef struct vect3_t
{
	vect3 v;
}vect3_t;

/*
homogeneous transformation matrix vectors (lol)
*/
typedef float vect4[4];
typedef struct vect4_t
{
	vect4 v;
}vect4_t;

/*
	spatial vectors
*/
typedef float vect6[6];
typedef struct vect6_t
{
	vect6 v;
}vect6_t;


extern const mat4_t mat4_t_Identity;
extern const mat3_t mat3_Identity;


vect6_t spatial_vect6_cross(vect6_t a, vect6_t b);
mat6_t spatial_cross_operator(vect6_t v);
vect3_t cross(vect3_t v_a, vect3_t v_b);
void cross_pbr(vect3_t* v_a, vect3_t* v_b, vect3_t* ret);
mat3_t skew(vect3_t vin);
vect6_t vect6_scale(vect6_t v_a, float scale);
vect4_t vect4_scale(vect4_t v_a, float scale);
vect3_t vect3_scale(vect3_t v_a, float scale);

vect6_t vect6_add(vect6_t v_a, vect6_t v_b);
vect4_t vect4_add(vect4_t v_a, vect4_t v_b);
vect3_t vect3_add(vect3_t v_a, vect3_t v_b);

float vect_dot(float* v1, float* v2, int n);
float vect_mag(float* v, int n);
void vect_normalize(float* v, int n);
vect3_t vect3_normalize(vect3_t v);

vect6_t zero_vect6();
vect6_t mat6_vect6_mult(mat6_t m, vect6_t v);
vect4_t mat4_t_vect4_mult(mat4_t m, vect4_t v);
vect3_t mat4_t_vect3_mult(mat4_t m, vect3_t v);	//untested
vect3_t mat3_vect3_mult(mat3_t m, vect3_t v);
mat6_t mat6_mult(mat6_t m1, mat6_t m2);
mat4_t mat4_t_mult(mat4_t m1, mat4_t m2);
void mat4_t_mult_pbr(mat4_t* m1, mat4_t* m2, mat4_t* ret);
mat3_t mat3_mult(mat3_t m1, mat3_t m2);
mat6_t mat6_add(mat6_t m1, mat6_t m2);
mat4_t mat4_t_add(mat4_t m1, mat4_t m2);
mat3_t mat3_add(mat3_t m1, mat3_t m2);
mat4_t mat4_t_I();
mat3_t mat3_I();
mat3_t Rx(float angle);
mat3_t Ry(float angle);
mat3_t Rz(float angle);
mat4_t ht_inverse(mat4_t hin);
void ht_inverse_ptr(mat4_t* hin, mat4_t* hout);
mat4_t Hx(float angle);
mat4_t Hy(float angle);
mat4_t Hz(float angle);
mat4_t Hscale(float scf);
mat3_t mat3_T(mat3_t in);
void matrix_vect_multiply(float ** M1, int rsize, float * v, int vsize, float * out);
mat6_t h4_to_X(mat4_t ha_b);
vect6_t c_change_spatial(mat4_t ha_b, vect6_t vb);

vect4_t vect3_to_vect4(vect3_t in);
vect3_t vect4_to_vect3(vect4_t in);

float Q_rsqrt(float number);

#endif