#ifndef GLFW_INTERFACE_H
#define GLFW_INTERFACE_H
#include <glad/glad.h>

#include <stdio.h>
#include <stdlib.h>

// Include GLFW
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "kinematics.h"
#include "objloader.hpp"
#include "vboindexer.hpp"
#include "texture.hpp"


#define CAM_NUM_FRAMES 4
//#define CAM_CHAIN_SIZE 3


typedef struct CamControlStruct
{
	uint8_t shiftlkey_down;
	uint8_t look_at_flag;
	char lock_in_flag; 
	char lkeydown;
	double horizontalAngle;
	double verticalAngle;
	double xyV;
	double jumpV;
	double telV;
	vect3_t cur_V;
	kinematic_chain CamRobot;
	
}CamControlStruct;

typedef struct VBOModelStruct
{
	// Get a handle for our "MVP" uniform
	GLuint MatrixID;
	GLuint ModelMatrixID;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
//	bool res = loadOBJ("MouseHead_VI.obj", vertices, uvs, normals);

	GLuint vertexbuffer;
	GLuint uvbuffer;
	GLuint normalbuffer;
	GLuint elementbuffer;
	//int iMouseHead = LoadVBO_OBJ(vertices, uvs, normals, &vertexbuffer, &uvbuffer, &normalbuffer, &elementbuffer);
	int isize;


	mat4_t HW_Model;
	glm::mat4 ModelMatrix;
	glm::mat4 MVP;
	
}VBOModelStruct;


//void init_cam(CamControlStruct * P, dh_entry * DH_table, float * q, float * qdot, ht_matrix * HTadj, ht_matrix * H0_idx);
//void init_cam(CamControlStruct * P, float * q, float * qdot, joint * j);
void init_cam(CamControlStruct * P, joint * j);



glm::mat4 ht_matrix_to_mat4_t(mat4_t H);
//glm::mat4 keyboard_cam_control(GLFWwindow* window, CamControlStruct * P, double fps);
glm::mat4 keyboard_cam_control(GLFWwindow* window, CamControlStruct * P, double fps, vect3_t otarg_w);

double RPF(double rv, double fps);
double TPF(double tv, double fps);

void initVBO_OBJ(VBOModelStruct * M, GLuint programID, const char * path);
void renderVBO_OBJ(VBOModelStruct * M, GLuint Texture, GLuint TextureID, glm::mat4 View, glm::mat4 Projection);


void RenderVBO_OBJ(glm::mat4 * MVP1, glm::mat4 * ModelMatrix1, GLuint MatrixID, GLuint ModelMatrixID, GLuint Texture,
	GLuint TextureID, GLuint * vertexbuffer, GLuint * uvbuffer, GLuint * normalbuffer, GLuint * elementbuffer, int size);

int LoadVBO_OBJ(std::vector<glm::vec3> vertices, std::vector<glm::vec2> uvs, std::vector<glm::vec3> normals,
	GLuint * vertexbuffer, GLuint * uvbuffer, GLuint * normalbuffer, GLuint *elementbuffer);





#endif