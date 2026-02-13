cmake_minimum_required(VERSION 4.2.3)

option(TABLEAU_DEMO_FETCH_DEPS "Fetch GLFW and GLAD for demos" ON)
option(TABLEAU_DEMO_USE_GLAD "Use GLAD function loader for demos" ON)

# Future extension point: optional VTK-based demo backend.
# GLAD2 generates a C target, so demos must enable C language.
enable_language(C)

# On macOS, OpenGL symbols are provided by the framework directly.
if(APPLE)
    set(TABLEAU_DEMO_USE_GLAD OFF CACHE BOOL "Use GLAD function loader for demos" FORCE)
endif()

if(TABLEAU_DEMO_FETCH_DEPS)
    include(FetchContent)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)

    if(TABLEAU_DEMO_USE_GLAD)
        FetchContent_Declare(
            glad
            GIT_REPOSITORY https://github.com/Dav1dde/glad.git
            GIT_TAG v2.0.8
            SOURCE_SUBDIR cmake
        )
        FetchContent_MakeAvailable(glad)
    endif()
else()
    find_package(glfw3 REQUIRED)
    if(TABLEAU_DEMO_USE_GLAD)
        find_package(glad CONFIG REQUIRED)
    endif()
endif()

find_package(OpenGL REQUIRED)

set(TABLEAU_GLFW_TARGET "")
if(TARGET glfw)
    set(TABLEAU_GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(TABLEAU_GLFW_TARGET glfw3::glfw)
else()
    message(FATAL_ERROR "Could not find a GLFW target (expected glfw or glfw3::glfw)")
endif()

set(TABLEAU_GLAD_TARGET "")
if(TABLEAU_DEMO_USE_GLAD)
    if(TARGET tableau_glad)
        set(TABLEAU_GLAD_TARGET tableau_glad)
    elseif(TARGET glad)
        set(TABLEAU_GLAD_TARGET glad)
    elseif(TARGET glad::glad)
        set(TABLEAU_GLAD_TARGET glad::glad)
    elseif(COMMAND glad_add_library)
        glad_add_library(tableau_glad REPRODUCIBLE API gl:core=3.3)
        set(TABLEAU_GLAD_TARGET tableau_glad)
    else()
        message(FATAL_ERROR "Could not configure a GLAD target (expected tableau_glad, glad, glad::glad, or glad_add_library())")
    endif()
endif()

set(TABLEAU_OPENGL_TARGET "")
if(TARGET OpenGL::GL)
    set(TABLEAU_OPENGL_TARGET OpenGL::GL)
elseif(TARGET OpenGL::OpenGL)
    set(TABLEAU_OPENGL_TARGET OpenGL::OpenGL)
else()
    message(FATAL_ERROR "Could not find an OpenGL target")
endif()

add_executable(three_body_demo
    src/main.cpp
    src/opengl_renderer.cpp
)

target_compile_features(three_body_demo PRIVATE cxx_std_17)
target_compile_definitions(three_body_demo PRIVATE GLFW_INCLUDE_NONE)
if(TABLEAU_DEMO_USE_GLAD)
    target_compile_definitions(three_body_demo PRIVATE TABLEAU_HAS_GLAD=1)
else()
    target_compile_definitions(three_body_demo PRIVATE TABLEAU_HAS_GLAD=0)
endif()
if(APPLE)
    target_compile_definitions(three_body_demo PRIVATE GL_SILENCE_DEPRECATION)
endif()

target_include_directories(three_body_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(three_body_demo
    PRIVATE
        tableau::core
        ${TABLEAU_GLFW_TARGET}
        ${TABLEAU_OPENGL_TARGET}
)
if(TABLEAU_DEMO_USE_GLAD)
    target_link_libraries(three_body_demo PRIVATE ${TABLEAU_GLAD_TARGET})
endif()

if(MSVC)
    target_compile_options(three_body_demo PRIVATE /W4)
else()
    target_compile_options(three_body_demo PRIVATE -Wall -Wextra -pedantic)
endif()

add_executable(black_hole_demo
    src/black_hole_main.cpp
    src/black_hole_renderer.cpp
)

target_compile_features(black_hole_demo PRIVATE cxx_std_17)
target_compile_definitions(black_hole_demo PRIVATE GLFW_INCLUDE_NONE)
if(TABLEAU_DEMO_USE_GLAD)
    target_compile_definitions(black_hole_demo PRIVATE TABLEAU_HAS_GLAD=1)
else()
    target_compile_definitions(black_hole_demo PRIVATE TABLEAU_HAS_GLAD=0)
endif()
if(APPLE)
    target_compile_definitions(black_hole_demo PRIVATE GL_SILENCE_DEPRECATION)
endif()

target_include_directories(black_hole_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(black_hole_demo
    PRIVATE
        tableau::core
        ${TABLEAU_GLFW_TARGET}
        ${TABLEAU_OPENGL_TARGET}
)
if(TABLEAU_DEMO_USE_GLAD)
    target_link_libraries(black_hole_demo PRIVATE ${TABLEAU_GLAD_TARGET})
endif()

if(MSVC)
    target_compile_options(black_hole_demo PRIVATE /W4)
else()
    target_compile_options(black_hole_demo PRIVATE -Wall -Wextra -pedantic)
endif()

add_executable(golf_demo
    src/golf_main.cpp
    src/golf_renderer.cpp
)

target_compile_features(golf_demo PRIVATE cxx_std_17)
target_compile_definitions(golf_demo PRIVATE GLFW_INCLUDE_NONE)
if(TABLEAU_DEMO_USE_GLAD)
    target_compile_definitions(golf_demo PRIVATE TABLEAU_HAS_GLAD=1)
else()
    target_compile_definitions(golf_demo PRIVATE TABLEAU_HAS_GLAD=0)
endif()
if(APPLE)
    target_compile_definitions(golf_demo PRIVATE GL_SILENCE_DEPRECATION)
endif()

target_include_directories(golf_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(golf_demo
    PRIVATE
        tableau::core
        ${TABLEAU_GLFW_TARGET}
        ${TABLEAU_OPENGL_TARGET}
)
if(TABLEAU_DEMO_USE_GLAD)
    target_link_libraries(golf_demo PRIVATE ${TABLEAU_GLAD_TARGET})
endif()

if(MSVC)
    target_compile_options(golf_demo PRIVATE /W4)
else()
    target_compile_options(golf_demo PRIVATE -Wall -Wextra -pedantic)
endif()
