#
# Morphologica incantations
#

# From CMAKE_SYSTEM work out which of __OSX__, __GLN__, __NIX__ are required
if(CMAKE_SYSTEM MATCHES Linux.*)
  set(OS_FLAG "-D__GLN__")
elseif(CMAKE_SYSTEM MATCHES BSD.*)
  set(OS_FLAG "-D__NIX__")
elseif(APPLE)
  set(OS_FLAG "-D__OSX__")
else()
  message(ERROR "Operating system not supported: " ${CMAKE_SYSTEM})
endif()

# morphologica uses c++-17 language features
set(CMAKE_CXX_STANDARD 17)

# Add the host definition to CXXFLAGS along with other switches, depending on OS/Compiler
if (APPLE)
  set(CMAKE_CXX_FLAGS "${OS_FLAG} -Wall -Wfatal-errors -g -O3")
else()
  # This assumes a gcc compiler (or a gcc mimic like Clang) // -Wfatal-errors
  set(CMAKE_CXX_FLAGS "${OS_FLAG} -Wall -g -O3 -Wno-unused-result -Wno-unknown-pragmas -march=native")
endif()

# Tell clang to be quiet about brace initialisers:
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
endif()

# Add OpenMP flags here, if necessary
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Additional GL compiler flags.
#
set(OpenGL_GL_PREFERENCE "GLVND") # Following `cmake --help-policy CMP0072`
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGL3_PROTOTYPES -DGL_GLEXT_PROTOTYPES")
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGL_SILENCE_DEPRECATION")
endif()

# Tell the program where the morph fonts are, to compile them into the binary
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMORPH_FONTS_DIR=\"\\\"${PROJECT_SOURCE_DIR}/morphologica/fonts\\\"\"")

# Find the libraries which will be needed
#find_package(OpenGL REQUIRED) # already found
#find_package(glfw3 3.2...3.4 REQUIRED) # already found
find_package(Freetype REQUIRED)

# Define collections of includes that have to be made for morphologica
set(MORPH_INC_CORE ${ARMADILLO_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR})
set(MORPH_INC_GL ${OPENGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS}) # not ${GLFW3_INCLUDE_DIR}
include_directories(${MORPH_INC_CORE} ${MORPH_INC_GL})
# Assuming that you installed morphologica in-tree (i.e. 'next to' schnakenberg.cpp).
set(MORPH_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/morphologica" CACHE PATH "The path to morphologica")
include_directories(BEFORE ${MORPH_INCLUDE_PATH}/include) # Allows GL3/gl3.h to be found
include_directories(BEFORE ${MORPH_INCLUDE_PATH})         # Allows morph/Header.h to be found

# The only MORPH_LIBS we need to link to are Freetype
set(MORPH_LIBS Freetype::Freetype)
#
# morphologica incantations complete
#

# We need to include libEyeRenderer3 headers
include_directories(BEFORE "../libEyeRenderer3")

# The OptiX way to create an executable
OPTIX_add_sample_executable (brainRenderer target_name brainrenderer.cpp OPTIONS -rdc true)

# Link CUDA, morphologica dependencies and libEyeRenderer3.so
target_link_libraries(${target_name} ${CUDA_LIBRARIES} ${MORPH_LIBS} EyeRenderer3)
