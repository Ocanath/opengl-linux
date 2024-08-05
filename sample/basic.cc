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
      //for (int i = 0; i < m->nu; i++)
      //{
      //    d->ctrl[2] = 0;
      //}
    //d->ctrl[2] =(40+ (sin(d->time)*0.5+0.5)*20)*3.14159265/180;
        vect3_t foot_xy_1;
        float h = 40; float w = 100;
        float period = 2.f;
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
            for (int leg = 0; leg < NUM_LEGS; leg++)
            {
                joint* j = &hexapod.leg[leg].chain[1];
                j->child->q += 10.0*PI/180.0;
            }

            int pld_idx = 0;
		    for(int leg = 0; leg < NUM_LEGS; leg++)
		    {
			    joint * j = &hexapod.leg[leg].chain[1];
			    for(int i = 0; i < 3; i++)
			    {
				    d->ctrl[pld_idx++] = j->q;
				    j = j->child;
			    }
		    }

      }
  }
  else
  {
    printf("fugg\r\n");
  }
}

// main function
int main(int argc, const char** argv) {
  printf("wascuzup bichs\r\n");

  // load and compile model
  char error[1000] = "Could not load binary model";
  // m = mj_loadXML("/home/admin/Psyonic/ability-hand-api/URDF/mujoco/abh_left_large.xml", 0, error, 1000);
  m = mj_loadXML("D:\\OcanathProj\\CAD\\hexapod\\mujoco\\hexapod.xml", 0, error, 1000);
    // m = mj_loadXML("/home/admin/OcanathProj/CAD/hexapod-cad/mujoco/hexapod.xml",0,error,1000);
  // m = mj_loadXML("/home/admin/OcanathProj/mujoco/model/humanoid/humanoid.xml", 0, error, 1000);
    if (!m) {
    mju_error("Load model error: %s", error);
  }
  // make data
  d = mj_makeData(m);

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

  mjcb_control = mycontroller;

  // run main loop, target real-time simulation and 60 fps rendering
  while (!glfwWindowShouldClose(window)) {
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    mjtNum simstart = d->time;
    while (d->time - simstart < 1.0/60.0) {
      mj_step(m, d);
    }

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
