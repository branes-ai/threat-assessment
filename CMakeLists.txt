cmake_minimum_required(VERSION 3.22)
project(threat-assessor
	DESCRIPTION "an embodied AI application that tracks video and assesses if there is a threat to be evaluated"
		VERSION "0.0.1"
		LANGUAGES C CXX ASM
		HOMEPAGE_URL "https://github.com/branes-ai/thread-assessment")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
	# correct __cplusplus variable setting
    # /Zc:__cplusplus
    # You need to compile with the /Zc:__cplusplus switch to see the updated value of the __cplusplus macro.
	# Microsoft tried updating the macro by default and discovered that a lot of code doesn't compile correctly
	# when they changed the value of __cplusplus.
	# They'll continue to require use of the /Zc:__cplusplus switch for all minor versions of MSVC in the 19.xx family.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

find_package(Boost)
if(Boost_FOUND)
	message(STATUS "Boost_INCLUDE_DIR" ${Boost_INCLUDE_DIR})
	include_directories(${Boost_INCLUDE_DIR})
endif(Boost_FOUND)

set(THREAT_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(THREAT_INSTALL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(THREAT_INSTALL_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")
set(THREAT_INSTALL_BIN_DIR "${PROJECT_SOURCE_DIR}/bin")

add_definitions(-D THREAT_ENABLE_TEST=ON)

# include file for common includes 
include_directories(${THREAT_INSTALL_INCLUDE_DIR})

enable_testing()
#include(CTest)

option(THREAT_CMAKE_TRACE "Trace cmake build file actions" ON)
option(THREAT_USE_FOLDERS "Enable solution folders in Visual Studio, disable for Express"   ON)
if (THREAT_USE_FOLDERS) 
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

####
# macro to read all cpp files in a directory
# and create a test target for that cpp file
macro (compile_all testing prefix folder)
    # cycle through the sources
    # For the according directories, we assume that each cpp file is a separate test
    # so, create a executable target and an associated test target
    foreach (source ${ARGN})
        get_filename_component (test ${source} NAME_WE)
        string(REPLACE " " ";" new_source ${source})
        set(test_name ${prefix}_${test})
        message(STATUS "Add test ${test_name} from source ${new_source}.")
        add_executable (${test_name} ${new_source})

        #add_custom_target(valid SOURCES ${SOURCES})
        set_target_properties(${test_name} PROPERTIES FOLDER ${folder})
        if (${testing} STREQUAL "true")
            if (THREAT_CMAKE_TRACE)
                message(STATUS "testing: ${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name}")
            endif()
            add_test(${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name})
        endif()
    endforeach (source)
endmacro (compile_all)

####
# macro to create an executable target consisting of all cpp files in a directory
# and create a test target for that cpp file
macro (compile_multifile_target testing test_name folder)
    message(STATUS "Add test ${test_name} from source folder ${folder}.")
    add_executable (${test_name} ${ARGN})

    #add_custom_target(valid SOURCES ${SOURCES})
    set_target_properties(${test_name} PROPERTIES FOLDER ${folder})
    if (${testing} STREQUAL "true")
        if (THREAT_CMAKE_TRACE)
            message(STATUS "testing: ${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name}")
        endif()
        add_test(${test_name} ${RUNTIME_OUTPUT_DIRECTORY}/${test_name})
    endif()
endmacro (compile_multifile_target)


# incorporate the specific targets that are contained in the source trees
add_subdirectory(src)
add_subdirectory(test)

