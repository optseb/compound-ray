#pragma once

#include <sutil/vec_math.h>
#include <GLFW/glfw3.h>
#include <bitset>

// A basic system for camera movement

enum class move_sense
{
    forward,
    backward,
    left,
    right,
    up,
    down,
    rotUp,
    rotDown,
    rotLeft,
    rotRight,
    zoomIn,
    zoomOut
};

class BasicController
{
public:
    float speed = 0.02f * 4.0f;
    float angularSpeed = 4.0f * M_PIf * 0.5f/180;

    // Returns true if an update is made
    bool ingestKeyAction (int32_t key, int32_t action)
    {
        bool output = false;
        std::bitset<12> mvst_cpy = this->move_state;
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            if (key == GLFW_KEY_W) {
                this->move_state.set (static_cast<int>(move_sense::forward));
            } else if (key == GLFW_KEY_A) {
                this->move_state.set (static_cast<int>(move_sense::left));
            } else if (key == GLFW_KEY_D) {
                this->move_state.set (static_cast<int>(move_sense::right));
            } else if (key == GLFW_KEY_S) {
                this->move_state.set (static_cast<int>(move_sense::backward));
            } else if (key == GLFW_KEY_P) {
                this->move_state.set (static_cast<int>(move_sense::up));
            } else if (key == GLFW_KEY_L) {
                this->move_state.set (static_cast<int>(move_sense::down));
            } else if (key == GLFW_KEY_UP) {
                this->move_state.set (static_cast<int>(move_sense::rotUp));
            } else if (key == GLFW_KEY_DOWN) {
                this->move_state.set (static_cast<int>(move_sense::rotDown));
            } else if (key == GLFW_KEY_LEFT) {
                this->move_state.set (static_cast<int>(move_sense::rotLeft));
            } else if (key == GLFW_KEY_RIGHT) {
                this->move_state.set (static_cast<int>(move_sense::rotRight));
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
        return (this->move_state ^ mvst_cpy).any();
    }

    float3 getMovementVector()
    {
        float3 output = make_float3(0.0f, 0.0f, 0.0f);
        if (this->move_state.test(static_cast<int>(move_sense::up))) {
            output += speed*UP;
            this->move_state.set(static_cast<int>(move_sense::up), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::down))) {
            output += speed*DOWN;
            this->move_state.set(static_cast<int>(move_sense::down), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::left))) {
            output += speed*LEFT;
            this->move_state.set(static_cast<int>(move_sense::left), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::right))) {
            output += speed*RIGHT;
            this->move_state.set(static_cast<int>(move_sense::right), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::forward))) {
            output += speed*FORWARD;
            this->move_state.set(static_cast<int>(move_sense::forward), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::backward))) {
            output += speed*BACK;
            this->move_state.set(static_cast<int>(move_sense::backward), false);
        }
        return output;
    }

    float getVerticalRotationAngle()
    {
        float out = 0.0f;
        if (this->move_state.test(static_cast<int>(move_sense::rotUp))) {
            out += angularSpeed;
            this->move_state.set(static_cast<int>(move_sense::rotUp), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::rotDown))) {
            out -= angularSpeed;
            this->move_state.set(static_cast<int>(move_sense::rotDown), false);
        }
        return out;
    }

    // Rightward is positive
    float getHorizontalRotationAngle()
    {
        float out = 0.0f;
        if (this->move_state.test(static_cast<int>(move_sense::rotLeft))) {
            out += angularSpeed;
            this->move_state.set(static_cast<int>(move_sense::rotLeft), false);
        }
        if (this->move_state.test(static_cast<int>(move_sense::rotRight))) {
            out -= angularSpeed;
            this->move_state.set(static_cast<int>(move_sense::rotRight), false);
        }
        return out;
    }

    // true if any part is set to true
    bool isActivelyMoving() { return this->move_state.any(); }

private:
    static constexpr float3 UP      = {0.0f,  1.0f, 0.0f};
    static constexpr float3 DOWN    = {0.0f, -1.0f, 0.0f};
    static constexpr float3 LEFT    = {-1.0f, 0.0f, 0.0f};
    static constexpr float3 RIGHT   = { 1.0f, 0.0f, 0.0f};
    static constexpr float3 FORWARD = {0.0f, 0.0f,  1.0f};
    static constexpr float3 BACK    = {0.0f, 0.0f, -1.0f};

    std::bitset<12> move_state;
};
