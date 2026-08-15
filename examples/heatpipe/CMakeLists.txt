include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(heatpipe_init heatpipe_init.cc)

target_link_libraries(
    heatpipe_init
    aslnum
    aslvtk
    asl
)