//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "libEyeRenderer.h"

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "MulticamScene.h"
#include "GlobalParameters.h"
#include "cameras/CompoundEyeDataTypes.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include <vector>
#include <stdexcept>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards
#ifdef BUFFER_TYPE_CUDA_DEVICE
  #define BUFFER_TYPE 0
#endif
#ifdef BUFFER_TYPE_GL_INTEROP
  #define BUFFER_TYPE 1
#endif
#ifdef BUFFER_TYPE_ZERO_COPY
  #define BUFFER_TYPE 2
#endif
#ifdef BUFFER_TYPE_CUDA_P2P
  #define BUFFER_TYPE 3
#endif

// By default, call the summing kernel within launchFrame so that it will always
// happen. It's potentially useful to call the summing kernel from inside getCameraData
// to profile its execution (seems to take 5 ms for summing/data transfer at 2048
// samples, dropping to 2 ms for 1024 samples on my i9/4080).
//
static constexpr bool sum_average_with_getCameraData = false;

MulticamScene scene;

globalParameters::LaunchParams*  d_params = nullptr;
globalParameters::LaunchParams   params   = {};
int32_t                 width    = 400;
int32_t                 height   = 400;

GLFWwindow* window = sutil::initUI( "Eye Renderer 3.0", width, height );
// outputBuffer would seem to be the width x height pixels of the window.
sutil::CUDAOutputBuffer<uchar4> outputBuffer(static_cast<sutil::CUDAOutputBufferType>(BUFFER_TYPE), width, height);
sutil::GLDisplay gl_display; // Stores the frame buffer to swap in and out

bool notificationsActive = true;

void initLaunchParams( const MulticamScene& scene ) {

    params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    params.frame = 0;
    params.lighting = false;

    const float loffset = scene.aabb().maxExtent();

    std::vector<Light::Point> lights(4);
    lights[0].color     = { 1.0f, 1.0f, 0.8f };
    lights[0].intensity = 5.0f;
    lights[0].position  = scene.aabb().center() + make_float3( loffset );
    lights[0].falloff   = Light::Falloff::QUADRATIC;
    lights[1].color     = { 0.8f, 0.8f, 1.0f };
    lights[1].intensity = 3.0f;
    lights[1].position  = scene.aabb().center() + make_float3( -loffset, 0.5f*loffset, -0.5f*loffset  );
    lights[1].falloff   = Light::Falloff::QUADRATIC;
    lights[2].color     = { 1.0f, 1.0f, 0.8f };
    lights[2].intensity = 5.0f;
    lights[2].position  = scene.aabb().center() + make_float3( 0.0f, 4.0f, -5.0f);
    lights[2].falloff   = Light::Falloff::QUADRATIC;
    lights[3].color     = { 1.0f, 1.0f, 0.8f };
    lights[3].intensity = 0.5f;
    lights[3].position  = scene.aabb().center() + make_float3( 1.0f, -6.0f, 0.0f);
    lights[3].falloff   = Light::Falloff::QUADRATIC;

    params.lights.count  = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.lights.data ),
                lights.size() * sizeof( Light::Point )
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( params.lights.data ),
                lights.data(),
                lights.size() * sizeof( Light::Point ),
                cudaMemcpyHostToDevice
                ) );

    params.miss_color   = make_float3( 0.1f );

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( globalParameters::LaunchParams ) ) );

    params.handle = scene.traversableHandle();
}


// Updates the params to acurately reflect the currently selected camera
void handleCameraUpdate( globalParameters::LaunchParams& params )
{
    //GenericCamera* camera  = scene.getCamera();

    // Make sure the SBT of the scene is updated for the newly selected camera before launch,
    // also push any changed host-side camera SBT data over to the device.
    scene.reconfigureSBTforCurrentCamera(false);
}

// Launch Optix threads to render a frame. Once this is done getCameraData() accesses the summed average values
void launchFrame( sutil::CUDAOutputBuffer<uchar4>& output_buffer, MulticamScene& scene )
{
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer        = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
                &params,
                sizeof( globalParameters::LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    if(scene.hasCompoundEyes() && scene.isCompoundEyeActive())
    {
      CompoundEye* camera = (CompoundEye*) scene.getCamera();
      // Launch the ommatidial renderer
      OPTIX_CHECK( optixLaunch(
                  scene.compoundPipeline(),
                  0,             // stream
                  reinterpret_cast<CUdeviceptr>( d_params ),
                  sizeof( globalParameters::LaunchParams ),
                  scene.compoundSbt(),
                  camera->getOmmatidialCount(),      // launch width
                  camera->getSamplesPerOmmatidium(), // launch height
                  1                                  // launch depth
                  ) );
      CUDA_SYNC_CHECK();
      params.frame++;// Increase the frame number
      camera->setRandomsAsConfigured();// Make sure that random stream initialization is only ever done once

      if constexpr (sum_average_with_getCameraData == false) {
        // After the compoundray pipeline, can call the sample-summing CUDA kernel here
        camera->averageRecordFrame();
        CUDA_SYNC_CHECK();
      }
    }

    // Launch render, but only if *required* as this can add slowness
    if(scene.enable_render_window == true) {
        OPTIX_CHECK( optixLaunch(
                         scene.pipeline(),
                         0,             // stream
                         reinterpret_cast<CUdeviceptr>( d_params ),
                         sizeof( globalParameters::LaunchParams ),
                         scene.sbt(),
                         width,  // launch width
                         height, // launch height
                         1//scene.getCamera()->samplesPerPixel // launch depth
                         ) );
    }

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void cleanup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params               ) ) );
    scene.cleanup();
}

//------------------------------------------------------------------------------
//
// API functions
//
//------------------------------------------------------------------------------
// General Running
//------------------------------------------------------------------------------
void setVerbosity(bool v)
{
  notificationsActive = v;
}
void loadGlTFscene(const char* filepath)
{
  loadScene(filepath, scene);
  scene.finalize();
  initLaunchParams(scene);
}
void setRenderSize(int w, int h)
{
  width = w;
  height = h;
  if(notificationsActive)
    std::cout<<"[PyEye] Resizing rendering buffer to ("<<w<<", "<<h<<")."<<std::endl;
  GLFWwindow* ctx = glfwGetCurrentContext();
  if (ctx != window) { glfwMakeContextCurrent (window); }
  outputBuffer.resize(width, height);
  glfwMakeContextCurrent (ctx);
}
double renderFrame(void)
{
  handleCameraUpdate(params);// Update the params to accurately reflect the currently selected camera

  auto then = std::chrono::steady_clock::now();
  launchFrame( outputBuffer, scene );
  CUDA_SYNC_CHECK();
  std::chrono::duration<double, std::milli> render_time = std::chrono::steady_clock::now() - then;

  if (notificationsActive) {
      std::cout<<"[PyEye] Rendered frame in "<<render_time.count()<<"ms."<<std::endl;
  }

  return(render_time.count());
}
void displayFrame(void)
{
  int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
  int framebuf_res_y = 0;   //
  glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );

  GLFWwindow* ctx = glfwGetCurrentContext();
  // Just in case the user just selected another window:
  if (ctx != window) { glfwMakeContextCurrent (window); }
  gl_display.display(
          outputBuffer.width(),
          outputBuffer.height(),
          framebuf_res_x,
          framebuf_res_y,
          outputBuffer.getPBO()
          );

  // Swap the buffer
  glfwSwapBuffers(window);
  glfwMakeContextCurrent (ctx);
}
void saveFrameAs(const char* ppmFilename)
{
  sutil::ImageBuffer buffer;
  buffer.data = outputBuffer.getHostPointer();
  buffer.width = outputBuffer.width();
  buffer.height = outputBuffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
  sutil::displayBufferFile(ppmFilename, buffer, false);
  if(notificationsActive)
    std::cout<<"[PyEye] Saved render as '"<<ppmFilename<<"'"<<std::endl;
}
unsigned char* getFramePointer(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame pointer..."<<std::endl;
  return (unsigned char*)outputBuffer.getHostPointer();
}
void getFrame(unsigned char* frame)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame..."<<std::endl;
  size_t displaySize = outputBuffer.width()*outputBuffer.height();
  for(size_t i = 0; i<displaySize; i++)
  {
    unsigned char val = (unsigned char)(((float)i/(float)displaySize)*254);
    frame[displaySize*3 + 0] = val;
    frame[displaySize*3 + 1] = val;
    frame[displaySize*3 + 2] = val;
  }
}
void stop(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Cleaning eye renderer resources."<<std::endl;
  sutil::cleanupUI(window);
  cleanup();
}

// C-level only
void * getWindowPointer()
{
  return (void*)window;
}

//------------------------------------------------------------------------------
// Camera Control
//------------------------------------------------------------------------------
size_t getCameraCount()
{
  return(scene.getCameraCount());
}
void nextCamera(void)
{
  scene.nextCamera();
}
size_t getCurrentCameraIndex(void)
{
  return(scene.getCameraIndex());
}
const char* getCurrentCameraName(void)
{
  return(scene.getCamera()->getCameraName());
}
void previousCamera(void)
{
  scene.previousCamera();
}
void gotoCamera(int index)
{
  scene.setCurrentCamera(index);
}
bool gotoCameraByName(char* name)
{
  scene.setCurrentCamera(0);
  for(auto i = 0u; i<scene.getCameraCount(); i++)
  {
    if(strcmp(name, scene.getCamera()->getCameraName()) == 0)
      return true;
    scene.nextCamera();
  }
  return false;
}
void setCameraPosition(float x, float y, float z)
{
  scene.getCamera()->setPosition(make_float3(x,y,z));
}
void getCameraPosition(float& x, float& y, float& z)
{
  const float3& camPos = scene.getCamera()->getPosition();
  x = camPos.x;
  y = camPos.y;
  z = camPos.z;
}
void setCameraLocalSpace(float lxx, float lxy, float lxz,
                         float lyx, float lyy, float lyz,
                         float lzx, float lzy, float lzz)
{
  scene.getCamera()->setLocalSpace(make_float3(lxx, lxy, lxz),
                                   make_float3(lyx, lyy, lyz),
                                   make_float3(lzx, lzy, lzz));
}
void rotateCameraAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateAround(angle,  make_float3(x,y,z));
}
void rotateCameraLocallyAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateLocallyAround(angle,  make_float3(x,y,z));
}
void rotateCamerasAround(float angle, float x, float y, float z)
{
  size_t cc = scene.getCameraCount();
  for (size_t i = 0; i < cc; ++i) {
    scene.getCamera()->rotateAround(angle,  make_float3(x,y,z));
    scene.nextCamera();
  }
}
void rotateCamerasLocallyAround(float angle, float x, float y, float z)
{
  size_t cc = scene.getCameraCount();
  for (size_t i = 0; i < cc; ++i) {
    scene.getCamera()->rotateLocallyAround(angle,  make_float3(x,y,z));
    scene.nextCamera();
  }
}
void translateCamera(float x, float y, float z)
{
  scene.getCamera()->move(make_float3(x, y, z));
}
void translateCameraLocally(float x, float y, float z)
{
  scene.getCamera()->moveLocally(make_float3(x, y, z));
}
void translateCamerasLocally(float x, float y, float z)
{
  // For each camera in scene:
  size_t cc = scene.getCameraCount();
  for (size_t i = 0; i < cc; ++i) {
    scene.getCamera()->moveLocally(make_float3(x, y, z));
    scene.nextCamera();
  } // at end should have looped back to original cam
}
void resetCameraPose()
{
  scene.getCamera()->resetPose();
}
void setCameraPose(float posX, float posY, float posZ, float rotX, float rotY, float rotZ)
{
  GenericCamera* c = scene.getCamera();
  c->resetPose();
  c->rotateAround(rotX, make_float3(1,0,0));
  c->rotateAround(rotY, make_float3(0,1,0));
  c->rotateAround(rotZ, make_float3(0,0,1));
  c->move(make_float3(posX, posY, posZ));
}

void getCameraData (std::vector<std::array<float, 3>>& cameraData)
{
  if (isCompoundEyeActive() == true) {

    if constexpr (sum_average_with_getCameraData == true) {
      // Alternative place to do the sample summing. Useful here, so that you can time
      // getCameraData() to work out how much time is taken to sum and transfer data to CPU
      ((CompoundEye*)scene.getCamera())->averageRecordFrame();
    }
    size_t omcount = ((CompoundEye*)scene.getCamera())->getOmmatidialCount();
    cameraData.resize (omcount);
    float3* _data = ((CompoundEye*)scene.getCamera())->getRecordFrame();
    for (size_t i = 0; i < omcount; ++i) {
      // copy _data[i] to cameraData[i] applying gamma correction
      // 1/2.2 = 0.45454545
      //cameraData[i] = { powf(_data[i].x, 1.0f/2.2f), powf(_data[i].y, 1.0f/2.2f), powf(_data[i].z, 1.0f/2.2f) };
      cameraData[i] = { _data[i].x, _data[i].y, _data[i].z };
    }
  } else {
    throw std::runtime_error ("Currently, getCameraData is implemented only for compound eye cameras");
  }
}

//------------------------------------------------------------------------------
// Ommatidial Camera Control
//------------------------------------------------------------------------------
bool isCompoundEyeActive(void)
{
  return scene.isCompoundEyeActive();
}

std::string getEyeDataPath()
{
  if (scene.isCompoundEyeActive()) { return scene.eye_data_path; }
  return std::string("");
}

void setCurrentEyeSamplesPerOmmatidium(int s)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->setSamplesPerOmmatidium(s);
  }
}
int getCurrentEyeSamplesPerOmmatidium(void)
{
  if(scene.isCompoundEyeActive())
  {
    return(((CompoundEye*)scene.getCamera())->getSamplesPerOmmatidium());
  }
  return -1;
}
void changeCurrentEyeSamplesPerOmmatidiumBy(int s)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(s);
  }
}
size_t getCurrentEyeOmmatidialCount(void)
{
  if(scene.isCompoundEyeActive())
  {
    return ((CompoundEye*)scene.getCamera())->getOmmatidialCount();
  }
  return 0;
}
void setOmmatidia(OmmatidiumPacket* omms, size_t count)
{
  // Break out if the current eye isn't compound
  if(!scene.isCompoundEyeActive())
    return;

  // First convert the OmmatidiumPacket list into an array of Ommatidium objects
  std::vector<Ommatidium> ommVector(count);
  for(size_t i = 0; i<count; i++)
  {
    OmmatidiumPacket& omm = omms[i];
    ommVector[i].relativePosition  = make_float3(omm.posX, omm.posY, omm.posZ);
    ommVector[i].relativeDirection = make_float3(omm.dirX, omm.dirY, omm.dirZ);
    ommVector[i].acceptanceAngleRadians = omm.acceptanceAngle;
    ommVector[i].focalPointOffset = omm.focalpointOffset;
  }

  // Actually set the new ommatidial structure
  ((CompoundEye*)scene.getCamera())->setOmmatidia(ommVector.data(), count);
}
const char* getCurrentEyeDataPath(void)
{
  if(scene.isCompoundEyeActive())
  {
    return ((CompoundEye*)scene.getCamera())->eyeDataPath.c_str();
  }
  return "\0";
}
void setCurrentEyeShaderName(char* name)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->setShaderName(std::string(name)); // Set the shader
    scene.reconfigureSBTforCurrentCamera(true); // Reconfigure for the new shader
  }
}

bool isInsideHitGeometry(float x, float y, float z, char* name)
{
  return scene.isInsideHitGeometry(make_float3(x, y, z), std::string(name), false);//notificationsActive);
}
float3 getGeometryMaxBounds(char* name)
{
  return scene.getGeometryMaxBounds(std::string(name));
}
float3 getGeometryMinBounds(char* name)
{
  return scene.getGeometryMinBounds(std::string(name));
}
