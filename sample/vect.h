/*
 * vect.h
 *
 *  Created on: Nov 21, 2021
 *      Author: Ocanath Robotman
 */

#ifndef INC_VECT_H_
#define INC_VECT_H_
#include <stdint.h>

typedef double vect3[3];
typedef struct vect3_t
{
	vect3 v;
}vect3_t;

/*
homogeneous transformation matrix vectors (lol)
*/
typedef double vect4[4];
typedef struct vect4_t
{
	vect4 v;
}vect4_t;


typedef double vect6[6];
typedef struct vect6_t
{
	vect6 v;
}vect6_t;

typedef double mat3[3][3];	//	Array wrapper for unambiguous pointer to 2d array
typedef struct mat3_t		//	Rotation matrices, skew symmetric matrices take this type
{
	mat3 m;
}mat3_t;

typedef double mat4[4][4];		//Array wrapper for unambiguous pointer to 2d array
typedef struct mat4_t			//homogeneous transformation matrices take this type. Allows working with arrays as a type directly (including returns) instead of pointers
{
	mat4 m;
}mat4_t;

typedef double mat6[6][6];
typedef struct mat6_t
{
	mat6 m;
}mat6_t;

/*********************Fixed point matrix-vect types*****************************/
typedef int32_t mat4_32b[4][4];
typedef struct mat4_32b_t
{
	mat4_32b m;
}mat4_32b_t;

typedef int32_t vect3_32b[3];
typedef struct vect3_32b_t
{
	vect3_32b v;
}vect3_32b_t;
typedef int32_t vect6_32b[6];
typedef struct vect6_32b_t
{
	vect6_32b v;
}vect6_32b_t;

typedef union {
	int8_t d8[sizeof(uint32_t) / sizeof(int8_t)];
	uint8_t u8[sizeof(uint32_t) / sizeof(uint8_t)];
	uint16_t u16[sizeof(uint32_t) / sizeof(uint16_t)];
	int16_t i16[sizeof(uint32_t) / sizeof(int16_t)];
	uint32_t u32;
	int32_t i32;
	float f32;	//sizeof(float) == sizeof(uint32_t) on this system
}u32_fmt_t;

//inline float asm_sqrt(float op1)
//{
//	float result;
//	asm volatile("vsqrt.f32 %0, %1" : "=w" (result) : "w" (op1) );
//	return result;
//}

mat4_t mat4_t_mult(mat4_t m1, mat4_t m2);
void mat4_t_mult_pbr(mat4_t * m1, mat4_t * m2, mat4_t * ret);
void cross_pbr(vect3_t * v_a, vect3_t * v_b, vect3_t * ret);
mat4_t Hz(double angle);
mat4_t Hy(double angle);
mat4_t Hx(double angle);
void ht_mat4_mult_pbr(mat4_t * m1, mat4_t * m2, mat4_t * ret);

void ht32_mult_pbr(mat4_32b_t * m1, mat4_32b_t * m2, mat4_32b_t * ret);
void cross32_pbr(vect3_32b_t * v_a, vect3_32b_t * v_b, vect3_32b_t * ret, int n);
void ht32_mult64_pbr(mat4_32b_t * m1, mat4_32b_t * m2, mat4_32b_t * ret, int n);

double vect_dot(double* v1, double* v2, int n);
vect6_t vect6_add(vect6_t v_a, vect6_t v_b);
/*Multiples vector v_a by scalar scale*/
vect6_t vect6_scale(vect6_t v_a, double scale);
vect3_t vect3_scale(vect3_t v_a, double scale);
void vect_normalize(double* v, int n);

mat4_32b_t Hy_nb(int32_t angle, int n);
mat4_32b_t Hx_nb(int32_t angle, int n);
mat4_32b_t Hz_nb(int32_t angle, int n);

void ht_inverse_ptr(mat4_t * hin, mat4_t * hout);

float Q_rsqrt(float number);

#endif /* INC_VECT_H_ */
