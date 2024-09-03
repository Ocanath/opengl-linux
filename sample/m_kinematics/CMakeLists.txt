set(MY_KINEMATICS
	dynahex.c
    dynahex.h
	hexapod_footpath.c
    hexapod_footpath.h
	kinematics.c
    kinematics.h
	sin-math.c
    sin-math.h
	trig_fixed.c
    trig_fixed.h
	vect.c
    vect.h
)
#target_include_directories(basic PRIVATE ${MY_KINEMATICS})
target_sources(basic PRIVATE ${MY_KINEMATICS})
target_compile_definitions(basic PRIVATE ${MY_KINEMATICS})
# target_sources(basic PRIVATE ${MY_KINEMATICS})
add_library(mkinematicslib SHARED ${MY_KINEMATICS})
