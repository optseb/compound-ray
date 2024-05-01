# A variable containing source code files to compile
set(brain_renderer_sources brainrenderer.cpp)

# Include the 'add_eye_renderer' macro from the eye renderer and
# then actually add the correct files to the source list so we can use
# eye renderer library commands.
include("../libEyeRenderer3/CMakeIncludes.cmake")
add_eye_renderer("../libEyeRenderer3" brain_renderer_sources)

OPTIX_add_sample_executable (brainRenderer target_name
  ${brain_renderer_sources}
  OPTIONS -rdc true
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})
