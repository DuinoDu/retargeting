#
# include("../cmake/rerun.cmake")
# target_link_libraries(target_name PRIVATE rerun_sdk)
#

cmake_minimum_required(VERSION 3.16...3.27)

if(ANDROID)
    set(RERUN_HOME /Users/duino/ws/rust_android/rerun)
    
    if(EXISTS ${RERUN_HOME}/build/android)
        set(BUILD_PREFIX "build/android")
    elseif(EXISTS ${RERUN_HOME}/build)
        set(BUILD_PREFIX "build")
    else()
        message(FATAL_ERROR "No build found in ${RERUN_HOME}")
    endif()

    set(RERUN_INCLUDE ${RERUN_HOME}/rerun_cpp/src)
    set(RERUN_LIB ${RERUN_HOME}/${BUILD_PREFIX}/rerun_cpp/librerun_sdk.a)
    set(RERUN_C_LIB ${RERUN_HOME}/target/aarch64-linux-android/release/librerun_c.a)

    set(ARROW_INCLUDE ${RERUN_HOME}/${BUILD_PREFIX}/rerun_cpp/arrow/include)
    set(ARROW_LIB ${RERUN_HOME}/${BUILD_PREFIX}/rerun_cpp/arrow/lib/libarrow.a)
    set(ARROW_LIB_DEP ${RERUN_HOME}/${BUILD_PREFIX}/rerun_cpp/arrow/lib/libarrow_bundled_dependencies.a)

    message(STATUS "RERUN_HOME=${RERUN_HOME}")
    message(STATUS "RERUN_INCLUDE=${RERUN_INCLUDE}")
    message(STATUS "RERUN_LIB=${RERUN_LIB}")

    include_directories(${RERUN_INCLUDE})
    include_directories(${ARROW_INCLUDE})

    add_library(rerun_c STATIC IMPORTED)
    set_target_properties(rerun_c PROPERTIES IMPORTED_LOCATION ${RERUN_C_LIB})

    add_library(rerun_cpp STATIC IMPORTED)
    set_target_properties(rerun_cpp PROPERTIES IMPORTED_LOCATION ${RERUN_LIB})

    add_library(arrow_lib STATIC IMPORTED)
    set_target_properties(arrow_lib PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
    
    add_library(arrow_dep_lib STATIC IMPORTED)
    set_target_properties(arrow_dep_lib PROPERTIES IMPORTED_LOCATION ${ARROW_LIB_DEP})

    add_library(rerun_sdk INTERFACE)
    target_link_libraries(rerun_sdk INTERFACE
        rerun_c
        rerun_cpp
        arrow_lib
        arrow_dep_lib
    )


else()
    include(FetchContent)
    FetchContent_Declare(rerun_sdk URL
        https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
    FetchContent_MakeAvailable(rerun_sdk)
endif()
