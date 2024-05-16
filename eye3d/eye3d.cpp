#include <iostream>
#include <vector>
#include <array>

// Currently, FPS is simply output very fast to stdout
//#define PROFILE_FPS 1
#ifdef PROFILE_FPS
# include <chrono>
#endif

// All for MulticamScene...
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
// and finally
#include "MulticamScene.h"

#include "libEyeRenderer.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include "BasicController.h"

// Include morph/Visual.h *after* glad and glfw header includes. I'm explicitly checking here so I
// can emit a compile error
#ifdef _glfw3_h_
# if defined __gl3_h_ || defined __gl_h_ // could instead check __glad_h_ here
#  include <morph/Visual.h>
# else
#  error "GL headers were not #included before morph/Visual.h as expected"
# endif
#else
# error "glfw3.h header was not #included before morph/Visual.h as expected"
#endif

// Our compound eye VisualModel class
#include "CompoundEyeVisual.h"

bool dirtyUI = true; // a flag to keep track of if the UI has changed in any way
BasicController controller;
bool simple_flared = true; // simple flared tubes in morphologica window

// scene exists at global scope in libEyeRenderer.so
extern MulticamScene scene;

static void keyCallback (GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    // Handle keypresses
    if (action == GLFW_PRESS) {
        if(key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
            glfwSetWindowShouldClose (window, true);
        } else if (key == GLFW_KEY_T) {
            // Toggle the morph view
            simple_flared = !simple_flared;
        } else {
            // Camera changing
            if (key == GLFW_KEY_N) {
                nextCamera();
            } else if (key == GLFW_KEY_B) {
                previousCamera();
            } else if(key == GLFW_KEY_PAGE_UP) {
                int csamp = getCurrentEyeSamplesPerOmmatidium();
                if (csamp < 32000) {
                    changeCurrentEyeSamplesPerOmmatidiumBy (csamp); // double
                } else {
                    // else graphics memory use will get very large
                    std::cout << "max allowed samples\n";
                }
            } else if (key == GLFW_KEY_PAGE_DOWN) {
                int csamp = getCurrentEyeSamplesPerOmmatidium();
                changeCurrentEyeSamplesPerOmmatidiumBy (-(csamp/2)); // halve
            } else if (key == GLFW_KEY_C) {
                saveFrameAs ("output.ppm");
            }

            dirtyUI = true;
        }
    }
    // Camera movement (mark the UI dirty if the controller has moved
    dirtyUI |= controller.ingestKeyAction (key, action);
}

static void windowSizeCallback (GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    setRenderSize (res_x, res_y);
    dirtyUI = true;
}

void printHelp()
{
    std::cout << "USAGE:\neye3d -f <path to gltf scene>" << std::endl << std::endl;
    std::cout << "\t-h\tDisplay this help information." << std::endl;
    std::cout << "\t-f\tPath to a gltf scene file (absolute or relative to current "
              << "working directory, e.g. './natural-standin-sky.gltf')." << std::endl;
}

int main (int argc, char* argv[])
{
    // Parse Inputs
    std::string path = "";
    for (int i=0; i<argc; i++) {
        std::string arg = std::string(argv[i]);
        if(arg == "-h") {
            printHelp();
            return 0;
        } else if(arg == "-f") {
            i++;
            path = std::string(argv[i]);
        }
    }

    if (path.empty()) {
        printHelp();
        return 1;
    }

    // Grab a pointer to the window
    auto window = static_cast<GLFWwindow*>(getWindowPointer());
    // Attach callbacks
    glfwSetKeyCallback (window, keyCallback);
    glfwSetWindowSizeCallback (window, windowSizeCallback);

    std::vector<std::array<float, 3>> ommatidiaData;
    std::vector<Ommatidium>* ommatidia = nullptr;
    try {
        // Turn off verbose logging
        setVerbosity (false);
        // Load the file
        std::cout << "Loading glTF file \"" << path << "\"..." << std::endl;
        loadGlTFscene (path.c_str());

        // Create a morphologica window to render the eye/sensor
        morph::Visual<> v (2000, 1200, "Morphologica graphics");
        v.setSceneTransZ (-0.8f);
        v.lightingEffects();

        morph::vec<float, 3> offset = { 0,0,0 };
        auto eyevm = std::make_unique<comray::CompoundEyeVisual<>> (offset, &ommatidiaData, ommatidia);
        v.bindmodel (eyevm);
        eyevm->finalize();
        auto eyevm_ptr = v.addVisualModel (eyevm);

        // The main loop
        size_t curr_eye_size = 0u;
        size_t last_eye_size = 0u;

#ifdef PROFILE_FPS
        using namespace std::chrono;
        using sc = std::chrono::steady_clock;
        sc::time_point t0 = sc::now(), t1 = sc::now();
#endif
        while (!glfwWindowShouldClose (window)) {
#ifdef PROFILE_FPS
            sc::duration t_d = t1 - t0;
            std::cout << "FPS " << 1000000.0 / static_cast<double>(duration_cast<microseconds>(t_d).count()) << "\n";
            t0 = sc::now();
#endif
            // Switch to morphologica context, poll, render and then release
            v.setContext();
            v.poll();
            eyevm_ptr->ommatidia = ommatidia;
            if (eyevm_ptr->ommatidia != nullptr) {
                curr_eye_size = eyevm_ptr->ommatidia->size();
                if (curr_eye_size != last_eye_size) {
                    eyevm_ptr->reinit();
                    last_eye_size = curr_eye_size;
                } else {
                    eyevm_ptr->updateColours(); // 4x faster to just updateColours
                }
            }
            v.render();
            v.releaseContext();

            // Set compound ray window context and poll for events
            glfwMakeContextCurrent (window);
            // For a fixed time-step model, it would be necessary to make a deterministic
            // wait-with-poll (see morph::Visual::wait(const double&) for an example of how) or
            // simply place in a fixed time sleep here
            glfwPollEvents();

            // Your brain model system may well NOT have a controller for moving the camera around,
            // or it may control the controller.
            if (controller.isActivelyMoving()) {
                float3 t = controller.getMovementVector(); // Local translation
                translateCamerasLocally (t.x, t.y, t.z);
                float va = controller.getVerticalRotationAngle();
                float vh = controller.getHorizontalRotationAngle();
                rotateCamerasLocallyAround (va, 1.0f, 0.0f, 0.0f);
                rotateCamerasAround (vh, 0.0f, 1.0f, 0.0f);
                dirtyUI = true;
            }

            // Do ray casting
            /* double ftime = */ renderFrame();
            // std::cout << "rendered frame in " << ftime << " ms\n";

            // Access data so that a brain model could be fed
            if (isCompoundEyeActive()) {
                getCameraData (ommatidiaData);
                ommatidia = &scene.m_ommVecs[scene.getCameraIndex()];
            }
            // For visual feedback, display in the GLFW window (if required)
            displayFrame();

            glfwMakeContextCurrent (nullptr);
#ifdef PROFILE_FPS
            t1 = sc::now();
#endif
        }

    } catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    // Stop goes at end *after* morph::Visual has gone out of scope.
    stop();

    return 0;
}
