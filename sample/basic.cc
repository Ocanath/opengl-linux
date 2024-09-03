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

dynahex_t hexapod;

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
  m = mj_loadXML("/home/admin/OcanathProj/CAD/hexapod-cad/mujoco/hexapod.xml", 0, error, 1000);
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
