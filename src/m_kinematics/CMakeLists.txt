set(MY_KINEMATICS
	dynahex.c
	hexapod_footpath.c
	kinematics.c
	sin-math.c
	trig_fixed.c
	vect.c
)

target_sources(mujoco PRIVATE ${MY_KINEMATICS})
