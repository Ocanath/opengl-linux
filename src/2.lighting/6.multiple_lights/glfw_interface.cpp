#include "glfw_interface.h"



void initVBO_OBJ(VBOModelStruct * M, GLuint programID, const char * path)
{
	M->MatrixID= glGetUniformLocation(programID, "MVP");
	M->ModelMatrixID = glGetUniformLocation(programID, "M");

	//read obj file. given function. not super awesome, but gets the job done
	bool res = loadOBJ(path, M->vertices, M->uvs, M->normals);
	
	M->isize = LoadVBO_OBJ(M->vertices, M->uvs, M->normals, &M->vertexbuffer, &M->uvbuffer, &M->normalbuffer, &M->elementbuffer);
}

void renderVBO_OBJ(VBOModelStruct * M, GLuint Texture, GLuint TextureID, glm::mat4 View, glm::mat4 Projection)
{
	
	M->ModelMatrix = ht_matrix_to_mat4_t(M->HW_Model);
	M->MVP = Projection * View * M->ModelMatrix;

	RenderVBO_OBJ(&M->MVP, &M->ModelMatrix, M->MatrixID, M->ModelMatrixID, Texture,
		TextureID, &M->vertexbuffer, &M->uvbuffer, &M->normalbuffer, &M->elementbuffer, M->isize);
}



//glm::mat4 ModelMatrix1 = ht_matrix_to_mat4_t(HW_MouseHead);
//glm::mat4 MVP1 = Projection * View * ModelMatrix1;
void RenderVBO_OBJ(glm::mat4 * MVP1, glm::mat4 * ModelMatrix1, GLuint MatrixID, GLuint ModelMatrixID, GLuint Texture,
	GLuint TextureID, GLuint * vertexbuffer, GLuint * uvbuffer, GLuint * normalbuffer, GLuint * elementbuffer, int size)
{
	// Send our transformation to the currently bound shader, 
	// in the "MVP" uniform
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &(*MVP1)[0][0]);
	glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &(*ModelMatrix1)[0][0]);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture);
	// Set our "myTextureSampler" sampler to user Texture Unit 0
	glUniform1i(TextureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, *vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
		);

	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, *uvbuffer);
	glVertexAttribPointer(
		1,                                // attribute
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);

	// 3rd attribute buffer : normals
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, *normalbuffer);
	glVertexAttribPointer(
		2,                                // attribute
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);

	// Index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *elementbuffer);

	// Draw the triangles !
//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glDrawElements(
		GL_TRIANGLES,      // mode
		size,    // count
		GL_UNSIGNED_SHORT,   // type
		(void*)0           // element array buffer offset
		);
	
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

}

int LoadVBO_OBJ(std::vector<glm::vec3> vertices, std::vector<glm::vec2> uvs, std::vector<glm::vec3> normals,
	GLuint * vertexbuffer, GLuint * uvbuffer, GLuint * normalbuffer, GLuint *elementbuffer)
{
	std::vector<unsigned short> indices;
	std::vector<glm::vec3> indexed_vertices;
	std::vector<glm::vec2> indexed_uvs;
	std::vector<glm::vec3> indexed_normals;
	indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);
	// Load it into a VBO

	glGenBuffers(1, vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, *vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);


	glGenBuffers(1, uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, *uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);


	glGenBuffers(1, normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, *normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

	// Generate a buffer for the indices as well

	glGenBuffers(1, elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
	return (int)indices.size();
}


/*
Initializes a cam struct. 
INPUT:
Properly sized arrays q,qdot,HTadj,H0_idx (size is CAMERA_ARR_SIZE), which are initialized into the players kinematic chain struct
CamControlStruct P, the player
OUTPUT:
Completely initialized cam control struct
*/
void init_cam(CamControlStruct * P, joint * j)
{
	int i;
	//glew stuff
	P->horizontalAngle = 0;
	P->verticalAngle = 0;
	P->lock_in_flag = 1;
	P->lkeydown = 0;
	//Default movement parameters
	//P->cur_V.u = 0;	P->cur_V.v = 0;	P->cur_V.w = 0;	//initial velocity
	for (i = 0; i < 3;i++)
		P->cur_V.v[i] = 0;
	P->xyV = 1.0;											//player movement velocity. if uninitalized, you'll zoom around
	P->jumpV = 1.0;										//
	P->telV = .2;											//
	//Robot kinematics stuff
	P->CamRobot.num_frames = CAM_NUM_FRAMES;
	P->CamRobot.j = j;
	//DH table definition
	//P->CamRobot.DH_Table[1].d = 0;			P->CamRobot.DH_Table[1].a = 0;			P->CamRobot.DH_Table[1].alpha = PI / 2;
	//P->CamRobot.DH_Table[2].d = 0;			P->CamRobot.DH_Table[2].a = 0;			P->CamRobot.DH_Table[2].alpha = -PI / 2; 
	//P->CamRobot.DH_Table[3].d = 0;			P->CamRobot.DH_Table[3].a = 0;			P->CamRobot.DH_Table[3].alpha = 0;

	//P->CamRobot.j[1].dh.d = 0;		P->CamRobot.j[1].dh.a = 0;		P->CamRobot.j[1].dh.alpha = PI/2;
	//P->CamRobot.j[2].dh.d = 0;		P->CamRobot.j[2].dh.a = 0;		P->CamRobot.j[2].dh.alpha = -PI / 2;
	//P->CamRobot.j[3].dh.d = 0;		P->CamRobot.j[3].dh.a = 0;		P->CamRobot.j[3].dh.alpha = 0;
	const dh_entry cambot_dh[] = 
	{
		{0, 0, 0},
		{0, 0, PI/2},
		{0, 0, -PI/2},
		{0, 0, 0}
	};

	 P->CamRobot.j[1].q = -PI / 2;
	 P->CamRobot.j[2].q = -PI / 2;
	 P->CamRobot.j[3].q = -PI / 2;
	
	//init fk
	//init_forward_kinematics_KC(&(P->CamRobot));
	 P->CamRobot.j[0].hb_i = mat4_t_I();
	 P->CamRobot.j[0].him1_i = mat4_t_I();
	init_forward_kinematics_dh(P->CamRobot.j, cambot_dh, P->CamRobot.num_frames-1);
}


glm::mat4 ht_matrix_to_mat4_t(mat4_t H)
{
	double HC_W_T[16];
	int r, c;
	for (r = 0; r < 4; r++)
	{
		for (c = 0; c < 4; c++)
		{
			HC_W_T[4 * c + r] = H.m[r][c];
		}
	}
	return glm::make_mat4(HC_W_T);


}

/*
	Takes in a desired rotational velocity in radians/sec, spits out the necessary rotation per frame
*/
double RPF(double rv, double fps)
{
	return rv / fps;
}

/*
	Takes in a desired translational velocity in world units/sec, spits out the necessary units/frame
*/
double TPF(double tv, double fps)
{
	return tv / fps;
}

glm::mat4 keyboard_cam_control(GLFWwindow* window, CamControlStruct * P, double fps, vect3_t otarg_w)
{
	double xpos, ypos;
	int xsize, ysize;
//	glfwGetWindowSize(window, &xsize, &ysize);
	xsize = 1024;
	ysize = 768;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (P->lock_in_flag == 1)
		glfwSetCursorPos(window, xsize / 2, ysize / 2);
	// Reset mouse position for next frame
	// Compute new orientation
	P->horizontalAngle += wrap_2pi(.003 * double(1024 / 2 - xpos));
	P->verticalAngle += wrap_2pi(.003 * double(768 / 2 - ypos));
	vect4_t move_v;
	move_v.v[0] = 0; move_v.v[1] = 0; move_v.v[2] = 0; move_v.v[3] = 1;
	char act = 0;
	//Forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		move_v.v[0] = -(float)TPF(P->xyV, fps);
		act = 1;
	}
	// Move backward
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		move_v.v[0] = (float)TPF(P->xyV, fps);
		act = 1;
	}
	// Strafe right
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		move_v.v[2] = -(float)TPF(P->xyV, fps);
		act = 1;
	}
	// Strafe left
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		move_v.v[2] = (float)TPF(P->xyV, fps);
		act = 1;
	}
	// Fly up
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
	{
		move_v.v[1] = (float)TPF(P->xyV, fps);
		act = 1;
	}
	// Fly down
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
	{
		move_v.v[1] = -(float)TPF(P->xyV, fps);
		act = 1;
	}
	// Display player parameters
	if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
	{
		vect3_t pos = h_origin(P->CamRobot.hw_b);
		//printf("coord = (%f, %f, %f) q = (%f, %f, %f)\r\n", pos.v[0], pos.v[1], pos.v[2], P->CamRobot.j[1].q, P->CamRobot.j[2].q, P->CamRobot.j[3].q);
		for(int i = 0; i < 3; i++)
			printf("Player.CamRobot.hw_b.m[%d][3] = %f;\r\n",i, pos.v[i]);
		for (int i = 1; i <= 2; i++)
			printf("Player.CamRobot.j[%d].q = fmod(%f + PI, 2 * PI) - PI;\r\n", i, P->CamRobot.j[i].q);
	}

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && P->CamRobot.hw_b.m[2][3] == -10)
	{
		move_v.v[1] = -(float)TPF(P->jumpV, fps);
		act = 1;
	}
	if ((glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) && P->lkeydown == 0 && glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		P->lkeydown = 1;
	if ((glfwGetKey(window, GLFW_KEY_L) == GLFW_RELEASE) && P->lkeydown == 1)
	{
		P->lkeydown = 0;
		P->lock_in_flag = !P->lock_in_flag;
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && P->shiftlkey_down == 0)
	{
		P->look_at_flag = (~P->look_at_flag) & 1;
		P->shiftlkey_down = 1;
	}
	else if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE)
		P->shiftlkey_down = 0;


	if (P->verticalAngle < -PI / 2)
		P->verticalAngle = -PI / 2;
	if (P->verticalAngle > PI / 2)
		P->verticalAngle = PI / 2;

	if (P->lock_in_flag == 0)
	{
	              //P->horizontalAngle = 0;
	               //P->verticalAngle = 0;
	}
	else
	{
		P->CamRobot.j[1].q = -PI / 2 - (float)P->horizontalAngle;
		P->CamRobot.j[2].q = -PI / 2 + (float)P->verticalAngle;
	}

	load_q(P->CamRobot.j->child);
	forward_kinematics(&P->CamRobot.j->hb_i, P->CamRobot.j->child);

	mat4_t HW_C3;
	mat4_t HW_C1;
	mat4_t HW_0;
	HW_0 = mat4_t_mult(P->CamRobot.hw_b, P->CamRobot.hb_0);		//HT_Multiply(P->CamRobot.HW_B, P->CamRobot.HB_0, &HW_0);
	HW_C1 = mat4_t_mult(HW_0, P->CamRobot.j[1].hb_i);	//HT_Multiply(HW_0, P->CamRobot.HT0_idx[1], &HW_C1);
	HW_C3 = mat4_t_mult(HW_0, P->CamRobot.j[3].hb_i); //HT_Multiply(HW_0, P->CamRobot.HT0_idx[3], &HW_C3);	//HW_B*HB_0*H0_3
	//^ in this case, 0=b
	if (P->look_at_flag)
	{
		vect3_t v_targ = vect3_add(h_origin(P->CamRobot.hw_b), vect3_scale(otarg_w, -1));
		vect3_t z = vect3_normalize(v_targ);
		vect3_t up = { { 0, 0, 1 } };
		vect3_t x = cross(up, z);
		x = vect3_normalize(x);
		vect3_t y = cross(z, x);
		y = vect3_normalize(y);

		for (int r = 0; r < 3; r++)
		{
			HW_C3.m[r][0] = x.v[r];
			HW_C3.m[r][1] = y.v[r];
			HW_C3.m[r][2] = z.v[r];
		}
		float za = move_v.v[0];	float xa = move_v.v[2];	float ya = move_v.v[1];
		vect4_t move_cf = { { xa, ya, za, 1 } };
		vect3_t move_v_W = vect4_to_vect3(mat4_t_vect4_mult(HW_C3, move_cf));
		vect3_t cv = vect3_add(vect3_scale(h_origin(P->CamRobot.hw_b), -1), move_v_W);
		for(int r = 0; r < 3; r++)
			P->CamRobot.hw_b.m[r][3] += cv.v[r];
	}
	else
	{
		vect4_t move_v_W;
		//HT_Point_Multiply(HW_C1, move_v, &move_v_W);
		move_v_W = mat4_t_vect4_mult(HW_C1, move_v);
		P->cur_V.v[0] = move_v_W.v[0] - P->CamRobot.hw_b.m[0][3];	//P->cur_V.u += move_v_W.v[0] - P->CamRobot.HW_B.H[0][3];
		P->cur_V.v[1] = move_v_W.v[1] - P->CamRobot.hw_b.m[1][3];	//P->cur_V.v += move_v_W.v[1] - P->CamRobot.HW_B.H[1][3];
		P->cur_V.v[2] = move_v_W.v[2] - P->CamRobot.hw_b.m[2][3];	//P->cur_V.w += move_v_W.v[2] - P->CamRobot.HW_B.H[2][3];

		P->CamRobot.hw_b.m[0][3] += P->cur_V.v[0];	//P->CamRobot.HW_B.H[0][3] += P->cur_V.u;
		P->CamRobot.hw_b.m[1][3] += P->cur_V.v[1];	//P->CamRobot.HW_B.H[1][3] += P->cur_V.v;
		P->CamRobot.hw_b.m[2][3] += P->cur_V.v[2];	//P->CamRobot.HW_B.H[2][3] += P->cur_V.w;
	}


	mat4_t HC3_W = ht_inverse(HW_C3);
	return ht_matrix_to_mat4_t(HC3_W);
}