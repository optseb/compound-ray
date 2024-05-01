#pragma once

#include <sutil/vec_math.h>
#include <GLFW/glfw3.h>

// A basic system for camera movement
class BasicController
{
public:
    float speed = 0.02f * 4.0f;
    float angularSpeed = 4.0f * M_PIf * 0.5f/180;

    // Returns true if an update is made
    bool ingestKeyAction(int32_t key, int32_t action)
    {
        bool output = false;

        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            if (key == GLFW_KEY_W) {
                output |= !forward;
                forward = true;
            } else if (key == GLFW_KEY_A) {
                output |= !left;
                left = true;
            } else if (key == GLFW_KEY_D) {
                output |= !right;
                right = true;
            } else if (key == GLFW_KEY_S) {
                output |= !backward;
                backward = true;
            } else if (key == GLFW_KEY_P) {
                output |= !up;
                up = true;
            } else if (key == GLFW_KEY_L) {
                output |= !down;
                down = true;
            } else if (key == GLFW_KEY_UP) {
                output |= !rotUp;
                rotUp = true;
            } else if (key == GLFW_KEY_DOWN) {
                output |= !rotDown;
                rotDown = true;
            } else if (key == GLFW_KEY_LEFT) {
                output |= !rotLeft;
                rotLeft = true;
            } else if (key == GLFW_KEY_RIGHT) {
                output |= !rotRight;
                rotRight = true;
            } else if (key == GLFW_KEY_END) {
                speed = speed * 0.5f;
                angularSpeed = angularSpeed * 0.5f;
                std::cout << "Speed reduced to " << speed << std::endl;
            } else if (key == GLFW_KEY_HOME) {
                speed = speed * 2.0f;
                angularSpeed = angularSpeed * 2.0f;
                std::cout << "Speed increased to " << speed << std::endl;
            }
        }
        return output;
    }

    float3 getMovementVector()
    {
        float3 output = make_float3(0.0f, 0.0f, 0.0f);
        if(up) { output += speed*UP; up = false; }
        if(down) { output += speed*DOWN; down = false; }
        if(left) { output += speed*LEFT; left = false; }
        if(right) { output += speed*RIGHT; right = false; }
        if(forward) { output += speed*FORWARD; forward = false; }
        if(backward) { output += speed*BACK; backward = false; }
        return output;
    }

    float getVerticalRotationAngle()
    {
        float out = 0.0f;
        if(rotUp) { out += angularSpeed; rotUp = false; }
        if(rotDown) { out -= angularSpeed; rotDown = false; }
        return out;
    }

    // Rightward is positive
    float getHorizontalRotationAngle()
    {
        float out = 0.0f;
        if(rotLeft) { out += angularSpeed; rotLeft = false; }
        if(rotRight) { out -= angularSpeed; rotRight = false; }
        return out;
    }

    // true if any part is set to true
    bool isActivelyMoving()
    {
        return forward || backward || left || right || up || down ||
        rotUp || rotDown || rotLeft || rotRight ||
        zoomIn || zoomOut;
    }

private:
    static constexpr float3 UP      = {0.0f,  1.0f, 0.0f};
    static constexpr float3 DOWN    = {0.0f, -1.0f, 0.0f};
    static constexpr float3 LEFT    = {-1.0f, 0.0f, 0.0f};
    static constexpr float3 RIGHT   = { 1.0f, 0.0f, 0.0f};
    static constexpr float3 FORWARD = {0.0f, 0.0f,  1.0f};
    static constexpr float3 BACK    = {0.0f, 0.0f, -1.0f};

    // Current key states
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool rotUp = false;
    bool rotDown = false;
    bool rotLeft = false;
    bool rotRight = false;
    bool zoomIn = false;
    bool zoomOut = false;
};
