// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "dynahex.h"
#include "sin-math.h"
#include "hexapod_footpath.h"
#include "kinematics.h"
#include "vect.h"

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


const dh_entry hexleg_dh[NUM_FRAMES_HEXLEG] = {
		{0,			0,					0},		//d,a,alpha
		{65.66f,	-53.2f,				PI/2},
		{29.00f,	-100.46602344f,		PI},
		{21.50f,	-198.31677025f,		0.f}
};

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


void forward_kinematics(mat4_t * hb_0, joint* f1_joint)
{
	if (f1_joint == NULL)
		return;

	joint * j = f1_joint;
	while(j != NULL)
	{
		//float sth = (float)sin((double)j->q);
		//float cth = (float)cos((double)j->q);
		float sth = j->sin_q;
		float cth = j->cos_q;

		mat4_t* r = &j->h_link;
		mat4_t * him1_i = &j->him1_i;	//specify lookup ptr first for faster loading

		him1_i->m[0][0] = cth * r->m[0][0] - r->m[1][0] * sth;
		him1_i->m[0][1] = cth * r->m[0][1] - r->m[1][1] * sth;
		him1_i->m[0][2] = cth * r->m[0][2] - r->m[1][2] * sth;
		him1_i->m[0][3] = cth * r->m[0][3] - r->m[1][3] * sth;
		him1_i->m[1][0] = cth * r->m[1][0] + r->m[0][0] * sth;
		him1_i->m[1][1] = cth * r->m[1][1] + r->m[0][1] * sth;
		him1_i->m[1][2] = cth * r->m[1][2] + r->m[0][2] * sth;
		him1_i->m[1][3] = cth * r->m[1][3] + r->m[0][3] * sth;
		him1_i->m[2][0] = r->m[2][0];
		him1_i->m[2][1] = r->m[2][1];
		him1_i->m[2][2] = r->m[2][2];
		him1_i->m[2][3] = r->m[2][3];

		j = j->child;
	}

	joint * parent = f1_joint;
	j = f1_joint;
	mat4_t_mult_pbr(hb_0, &j->him1_i, &j->hb_i);	//load hb_1.		hb_0 * h0_1 = hb_1
	while (j->child != NULL)
	{
		j = j->child;
		mat4_t_mult_pbr(&parent->hb_i, &j->him1_i, &j->hb_i);
		parent = j;
	}
}

/*
	Copies the contents of one mat4_t to the other. could use memcpy interchangably
*/
void copy_mat4_t(mat4_t * dest, mat4_t * src)
{
	for (int r = 0; r < 4; r++)
	{
		for (int c = 0; c < 4; c++)
		{
			dest->m[r][c] = src->m[r][c];
		}
	}
}


/*
	fast 2pi mod. needed for sin and cos FAST for angle limiting
 */
float fmod_2pi(float in)
{
	uint8_t aneg = 0;
	float in_eval = in;
	if(in < 0)
	{
		aneg = 1;
		in_eval = -in;
	}
	float fv = (float)((int)(in_eval*ONE_BY_TWO_PI));
	if(aneg == 1)
		fv = (-fv)-1;
	return in-TWO_PI*fv;
}


void init_forward_kinematics_dh(joint* j, const dh_entry* dh, int num_joints)
{
	for (int i = 1; i <= num_joints; i++)
	{
		float sin_alpha = sin(dh[i].alpha);
		float cos_alpha = cos(dh[i].alpha);
		/*Precomputed Hd*Ha*Halpha*/
		mat4_t link = {
			{
				{1,		0,				0,				dh[i].a},
				{0,		cos_alpha,		-sin_alpha,		0},
				{0,		sin_alpha,		cos_alpha,		dh[i].d},
				{0,		0,				0,				1}
			}
		};
		memcpy(&j[i].h_link, &link, sizeof(mat4_t));
		copy_mat4_t(&j[i].him1_i, &j[i].h_link);	//htheta of 0 is the identity
		j[i - 1].child = &j[i];
	}
	copy_mat4_t(&j[0].h_link, &j[0].hb_i);
	j[0].q = 0;
	j[num_joints].child = NULL;	//initialize the last node in the linked list to NULL
	for (int i = 1; i <= num_joints; i++)
		mat4_t_mult_pbr(&j[i - 1].hb_i, &j[i].him1_i, &j[i].hb_i);
}

/*
	Multiplies two mat4_t matrices, returns an entire structure. More wasteful
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

void init_dynahex_kinematics(dynahex_t * h)
{
	mat4_t hb_0_leg0 =
	{
			{
					{-1.f,	0,		0,		109.7858f},
					{0,		1.f,	0,		0},
					{0,		0,		-1.f,	0},
					{0,		0,		0,		1.f}
			}
	};
	const float angle_f = (2 * PI) / 6.f;
	for (int leg = 0; leg < NUM_LEGS; leg++)
	{
		joint * j = h->leg[leg].chain;
		j[0].him1_i = mat4_t_mult(Hz((float)leg * angle_f), hb_0_leg0);
		copy_mat4_t(&j[0].hb_i, &j[0].him1_i);

		init_forward_kinematics_dh(j, hexleg_dh, NUM_JOINTS_HEXLEG);
	}
}


/*pre-loads all sin_q and cos_q for the chain*/
void load_q(joint* chain_start)
{
	joint* j = chain_start;
	while (j != NULL)
	{
		j->sin_q = sin(j->q);
		j->cos_q = cos(j->q);

		j = j->child;
	}
}

//static vect3_t o_foottip_3 = {{ -15.31409f, -9.55025f, 0.f }};

void forward_kinematics_dynahexleg(dynahex_t* h)
{
	for (int leg = 0; leg < NUM_LEGS; leg++)
	{
		joint* j = &(h->leg[leg].chain[0]);
		load_q(j->child);
		forward_kinematics(&j->hb_i, j->child);

		//htmatrix_vect3_mult(&j[3].hb_i, &o_foottip_3, &h->leg[leg].ef_b);

		//calc_J_point(j, NUM_JOINTS_HEXLEG, h->leg[leg].ef_b);
	}
}





/*Get the distance between two 3 vectors*/
float dist_vect3(vect3_t * p0, vect3_t * p1)
{
	double sum_sq = 0.f;
	for (int i = 0; i < 3; i++)
	{
		float tmp = (p0->v[i] - p1->v[i]);
		tmp = tmp * tmp;
		sum_sq += (double)tmp;
	}
	//return (float)(sqrt(sum_sq));
	return sqrt(sum_sq);
}

/*
    Get the intersection of two circles in the x-y plane. Two solutions for valid inputs.
    Invalid inputs are those which result in the following cases:
        1. No intersection
        2. infinite intersections
        3. one solution
    It is simple to check for and reject these cases, but to save a little time and complexity we will
    assume correct inputs (all inputs for the 4bar linkage case are correct inputs)
 */
uint8_t get_intersection_circles(vect3_t * o0, float r0, vect3_t * o1, float r1, vect3_t solutions[2])
{
	float d = dist_vect3(o0, o1);
	//can do validity checks here (intersecting, contained in each other, equivalent) but they're unnecessary for our application

	//get some reused squares out of the way
	float r0_sq = r0 * r0;
	float r1_sq = r1 * r1;
	float d_sq = d * d;

	// solve for a
	float a = (r0_sq - r1_sq + d_sq) / (2 * d);

	// solve for h
	float h_sq = r0_sq - a * a;
	float h = sqrt(h_sq);

	float one_by_d = 1.f / d;
	// find p2
	vect3_t p2;
	p2.v[2] = 0.f;
	for (int i = 0; i < 2; i++)
		p2.v[i] = o0->v[i] + a * (o1->v[i] - o0->v[i]) * one_by_d;

	float t1 = h * (o1->v[1] - o0->v[1]) * one_by_d;
	float t2 = h * (o1->v[0] - o0->v[0]) * one_by_d;

	solutions[0].v[0] = p2.v[0] + t1;
	solutions[0].v[1] = p2.v[1] - t2;
	solutions[0].v[2] = 0.f;

	solutions[1].v[0] = p2.v[0] - t1;
	solutions[1].v[1] = p2.v[1] + t2;
	solutions[1].v[2] = 0.f;
	return 1;
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

/**/
float inverse_vect_mag(float* v, int n)
{
	float v_dot_v= 0.f;
	for (int i = 0; i < n; i++)
		v_dot_v += v[i] * v[i];
	return 1/sqrt(v_dot_v);
}


/**/
void vect_normalize(float* v, int n)
{
	float inv_mag = inverse_vect_mag(v,n);
	for (int i = 0; i < n; i++)
	{
		v[i] = v[i] * inv_mag;
	}
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


float atan2_approx(float sinVal, float cosVal)
{
	float abs_s = sinVal;
	if(abs_s < 0)
		abs_s = -abs_s;
	float abs_c = cosVal;
	if(abs_c < 0)
		abs_c = -abs_c;
	float min_v = abs_c;
	float max_v = abs_s;
	if(abs_s < abs_c)
	{
		min_v = abs_s;
		max_v = abs_c;
	}
	float a = min_v/max_v;
	float sv = a*a;
	float r = ((-0.0464964749 * sv + 0.15931422)*sv- 0.327622764) * sv * a + a;
	if(abs_s > abs_c)
		r = 1.57079637 -r;
	if(cosVal < 0)
		r = 3.14159274 - r;
	if(sinVal < 0)
		r = -r;
	return r;
}

float wrap_2pi(float in)
{
	return fmod_2pi(in + PI) - PI;
}


/*Note: this is further complicated by the fact that the q1 method of taking atan2_approx
 * is only valid for anchor points which are
 *
 */
void ik_closedform_hexapod(mat4_t * hb_0, joint * start, vect3_t * targ_b)
{
	mat4_t h0_b;
	ht_inverse_ptr(hb_0, &h0_b);

	vect3_t targ_0;
	htmatrix_vect3_mult(&h0_b, targ_b, &targ_0);

	/*A bit of geometric fuckery to account for the fact that the actual origin of the foot has a y offset from
	 * the angle formed at q={0,0,0}. A hack, basically. I think it works for
	 * any y offset you would encounter though, making it kinda useful.
	 *
	 * The q=0,0,0 y offset is currently -7.5mm
	 *
	 */
	vect3_t targ_0_offset;
	vect3_t z = { {0, 0, 1} };
	cross_pbr(&targ_0, &z, &targ_0_offset);	vect_normalize(targ_0_offset.v, 3);
	for (int i = 0; i < 3; i++)
		targ_0_offset.v[i] *= -7.5f;	//the y offset when the arm is in q={0,0,0}

	//final q1 result, load it into kinematic structure
	start->q = wrap_2pi(atan2_approx(targ_0.v[1] - targ_0_offset.v[1], targ_0.v[0] - targ_0_offset.v[0]) - PI);

	//do FK so we can express the target in frame 1 and do a 2 link planar arm solution
	forward_kinematics(hb_0, start);





	mat4_t h1_0;
	ht_inverse_ptr(&start->him1_i, &h1_0);

	vect3_t targ_1;
	htmatrix_vect3_mult(&h1_0, &targ_0, &targ_1);
	targ_1.v[2] = 0.f;	//set z value to 0, since it doesn't matter how we slide around z in our plane

	vect3_t zero = { {0,0,0} };
	vect3_t sols[2] = { 0 };
	get_intersection_circles(&zero, -hexleg_dh[2].a, &targ_1, -hexleg_dh[3].a, sols);

	int solidx = 1;
	joint* j = start->child;
	float theta1 = atan2_approx(sols[solidx].v[1], sols[solidx].v[0]);
	j->q = wrap_2pi(theta1 - PI);


	j = j->child;
	vect3_t vdif;
	for (int i = 0; i < 3; i++)
		vdif.v[i] = targ_1.v[i] - sols[solidx].v[i];
	float theta2 = atan2_approx(vdif.v[1], vdif.v[0]);
	j->q = wrap_2pi( PI-((PI-theta1)+theta2) );

}


void foot_path(float time, float h, float w, float period, vect3_t* v)
{
	float t = fmod_2pi(time*TWO_PI/period)/TWO_PI;	//easy way of getting time normalized to 0-1 using prior work, without needing standard lib fmod

	float p1 = 0.25f;	//half down, half up.
	float p2 = 1.f - p1;

	if (t >= 0 && t < p1)
	{
		t = (t - 0) / (p1 - 0);	//parametric function expects 0-1.

		v->v[0] = t * (w / 2);
		v->v[1] = 0;
		v->v[2] = 0;
	}
	else if(t >= p1 && t < p2)
	{
		t = 2.f * (t - p1) / (p2 - p1) - 1.f; // this function expects t = -1 to 1

		/*
			-1 to 1, except that the first couple of derivatives are lower at the ends.
			Consequence for smoother motion is that the footspeed is much higher in the middle of the motion
		*/
		for(int i = 0; i < 1; i++)	//the more iterations of this you run, the more like a step function this becomes.
			t = sin(HALF_PI * t);

		v->v[0] = -t * w / 2.f;
		//v->v[0] = -sin(t)*sin(t + HALF_PI) * 2.199f * (w / 2.f);
		v->v[1] = (-t*t + 1) * h;
		v->v[2] = 0;
	}
	else if(t >= p2 && t < 1)
	{
		t = (t - p2) / (1 - p2);

		v->v[0] = t * (w / 2) - (w / 2);
		v->v[1] = 0;
		v->v[2] = 0;
	}
	else
	{
		for (int i = 0; i < 3; i++)
			v->v[i] = 0;
	}
}

/*returns homogeneous transform mat4_t matrix which is the rotation 'analge' around the x axis */
mat4_t Hx(float angle)
{
	mat4_t ret;
	ret.m[0][0] = 1;	ret.m[0][1] = 0;				ret.m[0][2] = 0;				ret.m[0][3] = 0;
	ret.m[1][0] = 0;	ret.m[1][1] = cos(angle);	ret.m[1][2] = -sin(angle);	ret.m[1][3] = 0;
	ret.m[2][0] = 0;	ret.m[2][1] = sin(angle);	ret.m[2][2] = cos(angle);	ret.m[2][3] = 0;
	ret.m[3][0] = 0;	ret.m[3][1] = 0;				ret.m[3][2] = 0;				ret.m[3][3] = 1;
	return ret;
}
/*Loads rotation about coordinate. 0 = identity*/
mat4_t Hz(float angle)
{
	float cth = cos(angle);
	float sth = sin(angle);
	mat4_t r;
	r.m[0][0] = cth;		r.m[0][1] = -sth;		r.m[0][2] = 0;	r.m[0][3] = 0;
	r.m[1][0] = sth;		r.m[1][1] = cth;		r.m[1][2] = 0;	r.m[1][3] = 0;
	r.m[2][0] = 0;			r.m[2][1] = 0;			r.m[2][2] = 1;	r.m[2][3] = 0;
	r.m[3][0] = 0;			r.m[3][1] = 0;			r.m[3][2] = 0;	r.m[3][3] = 1;
	return r;
}


/*pass by reference htmatrix (special subset of mat4_t) and 3 vector)*/
void htmatrix_vect3_mult(mat4_t* m, vect3_t* v, vect3_t* ret)
{
	for (int r = 0; r < 3; r++)
	{
		float tmp = 0;
		for (int i = 0; i < 3; i++)
		{
			tmp += m->m[r][i] * v->v[i];
		}
		tmp += m->m[r][3];
		ret->v[r] = tmp;
	}
}

dynahex_t hexapod;


/*
* TODO TEST THIS FUNCTION
* Transforms a NORMALIZED quaternion to a rotation matrix
*/
mat4_t quat_to_mat4_t(vect4_t quat, vect3_t origin)
{
	mat4_t m;
	float q1 = quat.v[0];
	float q2 = quat.v[1];
	float q3 = quat.v[2];
	float q4 = quat.v[3];

	float qq1 = q1 * q1;
	float qq2 = q2 * q2;
	float qq3 = q3 * q3;
	float qq4 = q4 * q4;

	float q2q3 = q2 * q3;
	float q1q2 = q1 * q2;
	float q1q3 = q1 * q3;
	float q1q4 = q1 * q4;
	float q2q4 = q2 * q4;
	float q3q4 = q3 * q4;

	m.m[0][0] = qq1 + qq2 - qq3 - qq4;
	m.m[0][1] = 2.0f * (q2q3 - q1q4);
	m.m[0][2] = 2.0f * (q2q4 + q1q3);
	m.m[1][0] = 2.0f * (q2q3 + q1q4);
	m.m[1][1] = qq1 - qq2 + qq3 - qq4;
	m.m[1][2] = 2.0f * (q3q4 - q1q2);
	m.m[2][0] = 2.0f * (q2q4 - q1q3);
	m.m[2][1] = 2.0f * (q3q4 + q1q2);
	m.m[2][2] = qq1 - qq2 - qq3 + qq4;

	for (int r = 0; r < 3; r++)
		m.m[r][3] = origin.v[r];
	for (int c = 0; c < 3; c++)
		m.m[3][c] = 0;
	m.m[3][3] = 1.0f;
	return m;
}


void mycontroller(const mjModel * m, mjData* d)
{
  const char * modelname = &m->names[0];
  if(strcmp(modelname, "ability_hand") == 0)
  {
    // int qposidx[] = {7,9,11,13,15,16};
    for(int ch = 0; ch < 6; ch++)
    {
        double qdes = (20.0 + 50*(0.5+0.5*sin(d->time + (double)ch))) * 3.14159265/180;
        if(ch == 4)
          qdes = -qdes;
        d->ctrl[ch] = qdes;
    }
  }
  else if (strcmp(modelname, "hexapod") == 0)
  {  
    vect3_t foot_xy_1;
    float h = 40; float w = 100;
    float period = 4;
    foot_path(d->time, h, w, period, &foot_xy_1);

    vect3_t foot_xy_2;
    foot_path(d->time + period / 2.f, h, w, period, &foot_xy_2);


    /*Rotate and translate*/
    float forward_direction_angle = 0.f;	//y axis!
    mat4_t xrot = Hx(HALF_PI);
    mat4_t zrot = Hz(forward_direction_angle - HALF_PI);
    vect3_t tmp;
    htmatrix_vect3_mult(&xrot, &foot_xy_1, &tmp);
    htmatrix_vect3_mult(&zrot, &tmp, &foot_xy_1);	//done 1
    htmatrix_vect3_mult(&xrot, &foot_xy_2, &tmp);
    htmatrix_vect3_mult(&zrot, &tmp, &foot_xy_2);	//done 2


    //			int leg = 0;
    for (int leg = 0; leg < NUM_LEGS; leg++)
    {
        mat4_t lrot = Hz((TWO_PI / 6.f) * (float)leg);
        vect3_t o_motion_b = { { 270.f,0.f,-270.f } };
        htmatrix_vect3_mult(&lrot, &o_motion_b, &tmp);
        o_motion_b = tmp;

        vect3_t targ_b;
        for (int i = 0; i < 3; i++)
        {
            if (leg % 2 == 0)
            {
                targ_b.v[i] = foot_xy_1.v[i] + o_motion_b.v[i];
            }
            else
            {
                targ_b.v[i] = foot_xy_2.v[i] + o_motion_b.v[i];
            }
        }

        mat4_t* hb_0 = &hexapod.leg[leg].chain[0].him1_i;
        joint* start = &hexapod.leg[leg].chain[1];
        ik_closedform_hexapod(hb_0, start, &targ_b);
		forward_kinematics_dynahexleg(&hexapod);
        start->child->q = (-start->child->q + 1.39313132679);
        start->child->child->q = (start->child->child->q + 1.443345-HALF_PI);
        // printf("%f\r\n", d->ctrl[0]*180/PI);
    }

	
    
    // for(int leg = 0; leg < NUM_LEGS; leg++)
    // {
    //   joint * j = &hexapod.leg[leg].chain[1];
    //   j->q = 0;
	   //j->child->q = 1.39313132679;// wrap_2pi(0 + 1.72);
	   //j->child->child->q = 1.443345;// -(sin(d->time) * 0.5 + 0.5) * HALF_PI + 3.05;
    // }
	 //hexapod.leg[0].chain[1].child->q = sin(d->time);

    int ctrl_idx = 0;
    for(int leg = 0; leg < NUM_LEGS; leg++)
    {
      joint * j = &hexapod.leg[leg].chain[1];
      for(int i = 0; i < 3; i++)
      {
          d->ctrl[ctrl_idx++] = j->q;
          j = j->child;
      }
    }
  }
  else
  {
    printf("fugg\r\n");
  }
}


int get_htim1_i_from_linkname(mjModel* model, const char * name, mat4_t * mat4)
{
	int bodyID = mj_name2id(model, mjOBJ_BODY, name);
	vect4_t quat;
	for (int i = 0; i < 4; i++)
		quat.v[i] = model->body_quat[bodyID*4 + i];
	vect3_t pos;
	for (int i = 0; i < 3; i++)
		pos.v[i] = model->body_pos[bodyID*3 + i];
	*mat4 = quat_to_mat4_t(quat, pos);
	return 0;
}

int get_htworld_i_from_linkname(mjData* data, const char * name, mat4_t * mat4)
{
	int bodyID = mj_name2id(m, mjOBJ_BODY, name);
	vect4_t quat;
	for (int i = 0; i < 4; i++)
		quat.v[i] = data->xquat[bodyID*4 + i];
	vect3_t pos;
	for (int i = 0; i < 3; i++)
		pos.v[i] = data->xpos[bodyID*3 + i];
	*mat4 = quat_to_mat4_t(quat, pos);
	return 0;
}

double get_qpos_from_jointname(mjData * data, const char * name)
{
    int jointid = mj_name2id(m, mjOBJ_JOINT, name);
    jointid = m->jnt_qposadr[jointid];
	return d->qpos[jointid];
}

// main function
int main(int argc, const char** argv) {
  printf("wascuzup bichs\r\n");

  // load and compile model
  char error[1000] = "Could not load binary model";
  // m = mj_loadXML("/home/admin/Psyonic/ability-hand-api/URDF/mujoco/abh_left_large.xml", 0, error, 1000);
//    m = mj_loadXML("D:\\OcanathProj\\CAD\\hexapod\\mujoco\\hexapod.xml", 0, error, 1000);
    //m = mj_loadXML("/home/admin/OcanathProj/CAD/hexapod-cad/mujoco/hexapod.xml",0,error,1000);
  m = mj_loadXML("/home/redux/OcanathProj/hexapod-cad/mujoco/hexapod.xml", 0, error, 1000);
    if (!m) {
    mju_error("Load model error: %s", error);
  }
  // make data
  d = mj_makeData(m);

  mat4_t h2_3;
  get_htim1_i_from_linkname(m, "leg1_link3", &h2_3);
  printf("---------------\n");
  for(int r = 0; r < 4; r++)
  {
	for(int c = 0; c < 4; c++)
	{
		printf("%f ", h2_3.m[r][c]);
	}
	printf("\r\n");
  }
  printf("---------------\n");
  
  //todo: use the above model data to load up a kinematic tree
  //biggest problem: documentation does not make a DFS tree traversal very apparent.
  double o_axis[3] = { -0.0177556, 0.0988847, 0.0333596 };
  double angle_offset_q2 = atan2(-o_axis[0], o_axis[1] );
  printf("add %f to q2. in deg %f\r\n", angle_offset_q2, angle_offset_q2*180/3.14159265);
  double o_foottip[3] = { 0.027181, -0.212110, 0.031520 };
  double angle_offset_q3 = atan2(o_foottip[1], o_foottip[0]);
  printf("subtract %f from q3, in deg %f\r\n", angle_offset_q3, angle_offset_q3 * 180 / 3.14159265);

  printf("has %d dofs\r\n", m->nv);

  printf("model has %d dofs\r\n", m->nq);


  printf("model name: %s\r\n",&m->names[0]);
  printf("Joint names:\r\n");
  for(int i = 0; i < m->njnt; i++)
  {
    printf("    %s\r\n", &m->names[m->name_jntadr[i]]);
  }
  printf("Actuator names:\r\n");
  for(int i = 0; i < m->nu; i++)
  {
    printf("    %s\r\n", &m->names[m->name_actuatoradr[i]]);
  }

  if(strcmp( ((const char *)&m->names[0]), "hexapod") == 0  )
  {

  }

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  init_dynahex_kinematics(&hexapod);
  for (int leg = 0; leg < 6; leg++)
  {
      joint* j = hexapod.leg[leg].chain;
      j[1].q = 0;
      j[2].q = PI / 2;
      j[3].q = 0;
  }
	forward_kinematics_dynahexleg(&hexapod);

  mjcb_control = mycontroller;

  // run main loop, target real-time simulation and 60 fps rendering
  while (!glfwWindowShouldClose(window)) {
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    mjtNum simstart = d->time;
    while (d->time - simstart < 1.0/60.0) {
      //d->qpos[0] = 0;
      //d->qpos[1] = 0;
      //d->qpos[2] = 0.5;

      //d->qpos[3] = 0;
      //d->qpos[4] = 0;
      //d->qpos[5] = 0;
      //d->qpos[6] = 1;
      //
      mj_step(m, d);
    }


	//int bodyID = mj_name2id(m, mjOBJ_BODY, "leg1_link3");
	//vect4_t quat;
	//for (int i = 0; i < 4; i++)
	//	quat.v[i] = m->body_quat[bodyID*4 + i];
	//vect3_t pos;
	//for (int i = 0; i < 3; i++)
	//	pos.v[i] = m->body_pos[bodyID*3 + i];
	//mat4_t leg1h2_3 = quat_to_mat4_t(quat, pos);
	//int jointid = mj_name2id(m, mjOBJ_JOINT, "leg1_q2");
	//int qposid = m->jnt_qposadr[jointid];
	//mat4_t h2_3 = mat4_t_mult(Hz(d->qpos[qposid]), leg1h2_3);
	//
	//printf("%d, ", bodyID);
	//printf("h2_3 = \r\n");
	//for (int r = 0; r < 4; r++)
	//{
	//	printf("    ");
	//	for (int c = 0; c < 4; c++)
	//	{
	//		if (c < 3)
	//			printf("%f, ", h2_3.m[r][c]);
	//		else
	//			printf("%f", h2_3.m[r][c] * 1000.0);
	//	}
	//	printf("\r\n");
	//}
	//printf("\r\n");




	//int bodyID = mj_name2id(m, mjOBJ_BODY, "leg1_link3");
	//vect4_t quat;
	//for (int i = 0; i < 4; i++)
	//	quat.v[i] = d->xquat[bodyID*4 + i];
	//vect3_t pos;
	//for (int i = 0; i < 3; i++)
	//	pos.v[i] = d->xpos[bodyID*3 + i];
	//printf("%f, %f, %f\r\n", pos.v[0]*1000, pos.v[1]*1000, pos.v[2]*1000);

	

	mat4_t hw_base, hw_l1l1, hw_l1l2, hw_l1l3;
	get_htworld_i_from_linkname(d, "base", &hw_base);
	get_htworld_i_from_linkname(d, "leg1_link1", &hw_l1l1);
	get_htworld_i_from_linkname(d, "leg1_link1", &hw_l1l2);
	get_htworld_i_from_linkname(d, "leg1_link3", &hw_l1l3);
	mat4_t hbase_w;
	ht_inverse_ptr(&hw_base, &hbase_w);
	// printf("%f, %f, %f\r\n", hw_base.m[0][3]*1000, hw_base.m[1][3]*1000, hw_base.m[2][3]*1000);	
	double l1_q1 = get_qpos_from_jointname(d, "leg1_q1");
	double l1_q2 = get_qpos_from_jointname(d, "leg1_q2");
	double l1_q3 = get_qpos_from_jointname(d, "leg1_q3");
	mat4_t Hzq1 = Hz(l1_q1);
	mat4_t Hzq2 = Hz(l1_q2);
	mat4_t Hzq3 = Hz(l1_q3);

	mat4_t hb_link1, h1_link2, h2_link3;
	get_htim1_i_from_linkname(m, "leg1_link1", &hb_link1);
	get_htim1_i_from_linkname(m, "leg1_link2", &h1_link2);
	get_htim1_i_from_linkname(m, "leg1_link3", &h2_link3);
	mat4_t hb_1 = mat4_t_mult(hb_link1, Hzq1);
	//hb_1 = I*hb_1
	mat4_t h1_2 = mat4_t_mult(h1_link2, Hzq2);
	mat4_t hb_2 = mat4_t_mult(hb_1, h1_2);
	mat4_t h2_3 = mat4_t_mult(h2_link3, Hzq3);
	mat4_t hb_3 = mat4_t_mult(hb_2, h2_3);

	mat4_t h1_b, h2_b, h3_b;
	ht_inverse_ptr(&hb_3, &h3_b);
	ht_inverse_ptr(&hb_2, &h2_b);
	ht_inverse_ptr(&hb_1, &h1_b);

	mat4_t mj_hb_3 = mat4_t_mult(hbase_w, hw_l1l3);
	mat4_t test = mat4_t_mult(mj_hb_3, h3_b);
	printf("--------------------------------\n");
	for(int r = 0; r < 4; r++)
	{
		for( int c = 0; c < 4; c++)
		{
			printf("%f ", test.m[r][c]);
		}
		printf("\r\n");
	}
	printf("--------------------------------\n");


    // int base = mj_name2id(m,mjOBJ_BODY,"base");
    // double x = d->xpos[base*3];
    // double y = d->xpos[base*3+1];
    // double z = d->xpos[base*3+2];
    // printf("%f,%f,%f\r\n",x,y,z);

    //int actid = mj_name2id(m, mjOBJ_ACTUATOR, "l1_act2");
    //int jointid = mj_name2id(m, mjOBJ_JOINT, "leg1_q2");
    //jointid = m->jnt_qposadr[jointid];
    //printf("%f, %f\r\n",d->ctrl[actid], d->qpos[jointid]);
    


    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  //free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 1;
}
