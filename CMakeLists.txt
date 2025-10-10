cmake_minimum_required(VERSION 3.14)
project(robot-sim)

# Include FetchContent module
include(FetchContent)

FetchContent_Declare(
	mujoco
	GIT_REPOSITORY https://github.com/google-deepmind/mujoco
	GIT_TAG 3.3.6
)

FetchContent_MakeAvailable(mujoco)

# Create the main executable
add_executable(robot-sim
    src/main.cpp
)


target_link_libraries(robot-sim
	PRIVATE
	mujoco
)

target_include_directories(robot-sim
	PRIVATE
	${mujoco_SOURCE_DIR}
)
