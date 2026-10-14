cmake_minimum_required(VERSION 3.11)
project(manus_zmq_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MANUS SDK (proprietary, NOT redistributed with this repo): download the MANUS Core SDK
# for Linux and point MANUS_SDK_DIR at its ManusSDK folder (include/ManusSDK.h,
# lib/libManusSDK_Integrated.so). Default: manus/ManusSDK next to this server.
# See ../README.md.
if(NOT MANUS_SDK_DIR)
    if(DEFINED ENV{MANUS_SDK_DIR})
        set(MANUS_SDK_DIR "$ENV{MANUS_SDK_DIR}")
    else()
        set(MANUS_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ManusSDK")
    endif()
endif()
get_filename_component(MANUS_SDK_DIR "${MANUS_SDK_DIR}" ABSOLUTE)
if(NOT EXISTS "${MANUS_SDK_DIR}/include/ManusSDK.h" OR
   NOT EXISTS "${MANUS_SDK_DIR}/lib/libManusSDK_Integrated.so")
    message(FATAL_ERROR
        "MANUS SDK not found in '${MANUS_SDK_DIR}'.\n"
        "Expected include/ManusSDK.h and lib/libManusSDK_Integrated.so there. The SDK is "
        "proprietary and not shipped with this repo: download the MANUS Core SDK for Linux "
        "from MANUS, then either copy its ManusSDK folder to manus/ManusSDK or pass "
        "-DMANUS_SDK_DIR=/path/to/ManusSDK (or export MANUS_SDK_DIR). See manus/README.md.")
endif()
message(STATUS "MANUS SDK: ${MANUS_SDK_DIR}")

# Find ZMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Fetch nlohmann/json (header-only)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.10.0
    GIT_SHALLOW TRUE
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json cppzmq)

add_executable(manus_zmq_server
    src/main.cpp
    src/manus_zmq_server.cpp
    src/manus_calibration.cpp
)

target_include_directories(manus_zmq_server PRIVATE
    ${MANUS_SDK_DIR}/include
)

target_link_directories(manus_zmq_server PRIVATE
    ${MANUS_SDK_DIR}/lib
)

target_link_libraries(manus_zmq_server
    ManusSDK_Integrated
    cppzmq
    nlohmann_json::nlohmann_json
)

# RPATH so the binary finds the ManusSDK .so at runtime
set_target_properties(manus_zmq_server PROPERTIES
    INSTALL_RPATH "${MANUS_SDK_DIR}/lib"
    BUILD_RPATH "${MANUS_SDK_DIR}/lib"
)

install(TARGETS manus_zmq_server DESTINATION bin)
