include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(heatpipe heatpipe.cc)

target_link_libraries(
    heatpipe
    aslnum
    aslvtk
    asl
)