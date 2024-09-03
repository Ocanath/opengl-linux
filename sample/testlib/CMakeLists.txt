set(SROUSED
    testfcn.cc
)

add_library(myLib)
target_sources(myLib PRIVATE ${SROUSED})
target_include_directories(myLib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
