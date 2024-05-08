#include <iostream>
#include <vector>
#include <array>

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

//#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include "BasicController.h"

#define VISUAL_NO_GL_INCLUDE 1
//#include <morph/Visual.h>


bool dirtyUI = true; // a flag to keep track of if the UI has changed in any way
BasicController controller;

// scene exists at global scope in libEyeRenderer.so
extern MulticamScene scene;

static void keyCallback (GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    // Handle keypresses
    if (action == GLFW_PRESS) {
        if(key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
            glfwSetWindowShouldClose (window, true);
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
    std::cout << "USAGE:\nbrainrenderer -f <path to gltf scene>" << std::endl << std::endl;
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

//    morph::Visual<> v (1000, 750, "Graphics");
//    v.addLabel ("Graphics", {0,0,0});

    // Grab a pointer to the window
    auto window = static_cast<GLFWwindow*>(getWindowPointer());
    // Attach callbacks
    glfwSetKeyCallback (window, keyCallback);
    glfwSetWindowSizeCallback (window, windowSizeCallback);

    std::vector<std::array<float, 3>> ommatidiaData;
    try {
        // Turn off verbose logging
        setVerbosity (false);
        // Load the file
        std::cout << "Loading glTF file \"" << path << "\"..." << std::endl;
        loadGlTFscene (path.c_str());

        // The main loop
        while (!glfwWindowShouldClose (window)) {

            // For a fixed time-step model, it would be necessary to make a deterministic
            // wait-with-poll (see morph::Visual::wait(const double&) for an example of how) or
            // simply place in a fixed time sleep here
            glfwPollEvents();

            //v.render();

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

                // Also need ommatidial info (in scene.m_ommVecs; scene exists at global scope)
                std::cout << "camIndex " << scene.getCameraIndex() << "; size of ommatidiaData is " << ommatidiaData.size()
                          << ", size of m_ommVecs is " << scene.m_ommVecs.size()
                          << " and size of our ommVec is " << scene.m_ommVecs[scene.getCameraIndex()].size() << std::endl;


#if 0
                for (auto omm : scene.m_ommVecs[scene.getCameraIndex()]) {
                    std::cout << "coord (" << omm.relativePosition.x
                              << "," << omm.relativePosition.y << "," << omm.relativePosition.z << ")"
                              << " angle (" << omm.relativeDirection.x
                              << "," << omm.relativeDirection.y << "," << omm.relativeDirection.z << ")"
                              << ") acceptance: " << omm.acceptanceAngleRadians << "\n";
                }
#endif
            }
            // For visual feedback, display in the GLFW window (if required)
            displayFrame();
        }
        stop();

    } catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
