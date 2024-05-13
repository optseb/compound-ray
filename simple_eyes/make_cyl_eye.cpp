// Make a cylindrical eye based on eye-specification.txt
#include <morph/mathconst.h>
#include <cmath>
#include <iostream>

int main()
{
    using mc = morph::mathconst<float>;

    constexpr int n_per_ring = 100;
    constexpr int n_rings = 32;
    // ring radius in mm
    constexpr float ring_rad = 0.2f;
    // Angle between elements in the ring
    constexpr float element_angle = mc::two_pi / n_per_ring;
    // Compute to be same as ommatidium to ommatidium in the ring
    constexpr float ring_to_ring = (mc::pi * ring_rad * 2.0f) / static_cast<float>(n_per_ring);

    // A truly cylindrical array isn't much use as an eye. If you want that, set this to
    // 0.0f. Otherwise, give an angle in radians that is the vertical field of view for the
    // cyl. array
    constexpr float vertical_array_angle = mc::pi_over_2;
    constexpr float vertical_el_angle = vertical_array_angle / (n_rings - 1);
    constexpr float start_angle = -vertical_array_angle / 2.0f;

    // Position
    float xp = 0.0f;
    float yp = 0.0f;
    float zp = 0.0f;
    // Direction
    float xd = 0.0f;
    float yd = 0.0f;
    float zd = 0.0f;

    // Acceptance angle (rad)
    constexpr float a = element_angle;
    // Focal point
    constexpr float f = 0.0f;

    for (int r = 0; r < n_rings; ++r) {
        zp = ring_to_ring * r;
        zd = std::sin (start_angle + (r-1) * vertical_el_angle);
        for (int el = 0; el < n_per_ring; ++el) {
            xp = ring_rad * std::cos (el * element_angle);
            yp = ring_rad * std::sin (el * element_angle);

            xd = std::cos (el * element_angle);
            yd = std::sin (el * element_angle);

            std::cout << xp << " " << yp << " " << zp << " "
                      << xd << " " << yd << " " << zd << " "
                      << a << " " << f << "\n";
        }
    }

    return 0;
}
