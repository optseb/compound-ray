#include <iostream>
#include <sutil/sutil.h>
#include "libEyeRenderer.h"
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include "BasicController.h"

bool dirtyUI = true; // a flag to keep track of if the UI has changed in any way
BasicController controller;

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

    // Grab a pointer to the window
    auto window = static_cast<GLFWwindow*>(getWindowPointer());
    // Attach callbacks
    glfwSetKeyCallback (window, keyCallback);
    glfwSetWindowSizeCallback (window, windowSizeCallback);

    try {
        // Turn off verbose logging
        setVerbosity (false);
        // Load the file
        std::cout << "Loading glTF file \"" << path << "\"..." << std::endl;
        loadGlTFscene (path.c_str());

        // The main loop
        while (!glfwWindowShouldClose (window)) {
            glfwPollEvents(); // Check if anything's happened, user-input-wise.

            if (controller.isActivelyMoving()) {
                float3 t = controller.getMovementVector();// Local translation
                translateCamerasLocally (t.x, t.y, t.z);
                float va = controller.getVerticalRotationAngle();
                float vh = controller.getHorizontalRotationAngle();
                rotateCamerasLocallyAround (va, 1.0f, 0.0f, 0.0f);
                rotateCamerasAround (vh, 0.0f, 1.0f, 0.0f);
                dirtyUI = true;
            }

            if (dirtyUI || isCompoundEyeActive()) {
                /*double ftime =*/ renderFrame();
                // std::cout << "rendered frame in " << ftime << " ms\n";
                displayFrame();
                dirtyUI = false; // Comment this out to force constant re-rendering
            }
        }
        stop();

    } catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}