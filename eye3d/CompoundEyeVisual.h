//
// A visualmodel to render a compound-ray compound eye.
//

#pragma once

#include <morph/vec.h>
#include <morph/VisualModel.h>
#include <morph/mathconst.h>
#include <morph/gl/version.h>
#include <array>
#include <vector>

#include "cameras/CompoundEyeDataTypes.h"

namespace comray {

    //! This class creates the vertices for a cylindrical 'rod' in a 3D scene.
    template<int glver = morph::gl::version_4_1>
    class CompoundEyeVisual : public morph::VisualModel<glver>
    {
    public:
        CompoundEyeVisual() { this->mv_offset = {0.0, 0.0, 0.0}; }

        //! Initialise with offset, start and end coordinates, radius and a single colour.
        CompoundEyeVisual(const morph::vec<float, 3> _offset,
                          std::vector<std::array<float, 3>>* _ommData,
                          std::vector<Ommatidium>* _ommatidia)
        {
            this->init (_offset, _ommData, _ommatidia);
        }

        ~CompoundEyeVisual () {}

        void init (const morph::vec<float, 3> _offset,
                   std::vector<std::array<float, 3>>* _ommData,
                   std::vector<Ommatidium>* _ommatidia)
        {
            this->mv_offset = _offset;
            this->viewmatrix.translate (this->mv_offset);
            this->ommData = _ommData;
            this->ommatidia = _ommatidia;
        }

        void updateColours()
        {
            if (ommData == nullptr) { return; }
            if (ommData->empty()) { return; }
            size_t n_verts = this->vertexColors.size(); // should be tube_vertices * n_omm
            if (n_verts == 0u) { return; } // model doesn't exist yet

            this->vertexColors.clear(); // Could re-write not clear/push
            size_t n_omm = ommData->size();

            int num_vertices = disc_vertices;
            if (this->show_cones == true) {
                num_vertices = cone_vertices + disc_vertices;
            } // else num_vertices = disc_vertices;

            // 3 colours, n_omm tubes, cone_vertices vertices per cone.
            if (n_verts != 3u * n_omm * static_cast<unsigned int>(num_vertices)) {
                throw std::runtime_error ("CompoundEyeVisual: n_verts/n_omm sizes mismatch!");
            }

            // Could test on: if (this->focal_point_sum > 0.0f) { // if the number of vertices were
            // different, but I chose cones in both cases

            // Re-colour cones built from a focal point offset and acceptance angle
            for (size_t i = 0u; i < n_omm; ++i) {
                // Update the 3 RGB values in vertexColors tube_vertices times
                int j = 0;
                for (; j < num_vertices; ++j) {
                    this->vertex_push ((*ommData)[i], this->vertexColors);
                }
            }

            // Lastly, this call copies vertexColors (etc) into the OpenGL memory space
            this->reinit_colour_buffer();
        }

        //! Initialize vertex buffer objects and vertex array object.
        void initializeVertices()
        {
            this->vertexPositions.clear();
            this->vertexNormals.clear();
            this->vertexColors.clear();
            this->indices.clear();

            // Sanity check our data pointers and return or throw
            if (ommData == nullptr || ommatidia == nullptr) { return; }
            if (ommatidia != nullptr && ommatidia->empty()) { return; }
            if (ommData != nullptr && ommData->empty()) { return; }
            if (ommData->size() != ommatidia->size()) {
                throw std::runtime_error ("sizes mismatch!");
            }

            // Draw ommatidia
            size_t n_omm = ommData->size();

            // First find out if all focal points are 0
            this->focal_point_sum = 0.0f;
            for (size_t i = 0u; i < n_omm; ++i) {
                this->focal_point_sum += (*ommatidia)[i].focalPointOffset;
            }

            if (this->focal_point_sum > 0.0f) {
                // We have focal points, so draw with the relativePosition representing the centre
                // of the ommatidial lens - the base of a cone - which then extends back to the cone
                // tip, which can be thought of as the location of the ommatidial 'sensor'
                for (size_t i = 0u; i < n_omm; ++i) {
                    // Ommatidia colour, position/shape
                    std::array<float, 3> colour = (*ommData)[i];
                    float3 rpos = (*ommatidia)[i].relativePosition;
                    float3 rdir = (*ommatidia)[i].relativeDirection;
                    float angle = (*ommatidia)[i].acceptanceAngleRadians;
                    float focal_point = (*ommatidia)[i].focalPointOffset;
                    morph::vec<float, 3> pos = { rpos.x, rpos.y, rpos.z };
                    morph::vec<float, 3> dir = { rdir.x, rdir.y, rdir.z };
                    dir.renormalize();
                    // Tip of cone is 'behind' the position of the ommatidial face/lens
                    morph::vec<float, 3> ommatidial_detector_point = pos - dir * focal_point;
                    // work out radius from acceptance angle and focal_point
                    float radius = focal_point * std::tan (angle / 2.0f);
                    // The disc
                    this->computeTube (this->idx, pos, pos + (0.1f * radius * dir), colour, colour, radius, tube_faces);
                    if (this->show_cones == true) {
                        // Colour comes from ommData. ringoffset is 1.0f
                        this->computeCone (this->idx, pos, ommatidial_detector_point, 0.0f, colour, radius, tube_faces);
                    }
                    std::cout << std::endl;
                }
            } else {
                // All our focal_points are 0. Don't have focal point offset to help define our
                // cones, only acceptance angle. Use manually specified tube_length (or computed
                // radius) to figure out the size of a cone, whose tip is the location of the
                // ommatidial sensor AND the centre of the ommatidial lens
                for (size_t i = 0u; i < n_omm; ++i) {
                    std::array<float, 3> colour = (*ommData)[i];
                    float3 rpos = (*ommatidia)[i].relativePosition;
                    float3 rdir = (*ommatidia)[i].relativeDirection;
                    float angle = (*ommatidia)[i].acceptanceAngleRadians;
                    // pos will be the tip of the cone in this case, and the centre of the disc
                    morph::vec<float, 3> pos = { rpos.x, rpos.y, rpos.z };
                    morph::vec<float, 3> dir = { rdir.x, rdir.y, rdir.z };
                    dir.renormalize();
                    // do a cone
                    morph::vec<float, 3> ommatidial_cone_pos = pos + dir * this->cone_length;
                    float ringoffset = 0.0f;
                    // work out radius from acceptance angle and focal_point
                    float radius = this->disc_width / 2.0f; // will be negative if not set
                    if (radius < 0.0f) { // fall back to using cone_length
                        radius = this->cone_length * std::tan (angle / 2.0f);
                    }
                    // Show a disc. Use disc_width, or if it is -ve, cone_length and computed radius
                    this->computeTube (this->idx, pos, pos - (0.1f * radius * dir), colour, colour, radius, tube_faces);
                    // And optionally a cone
                    if (this->show_cones == true) {
                        this->computeCone (this->idx, ommatidial_cone_pos, pos, ringoffset, colour, radius, tube_faces);
                    }
                }
            }

        }

        // Visualize in two modes "disc" mode, showing just a 2D disc for each ommatidium and
        // disc+cone mode, where the acceptance angle is displayed too. Runtime switchable.
        bool show_cones = false;
        // The colours detected by each ommatidium
        std::vector<std::array<float, 3>>* ommData = nullptr;
        // The position and orientation of each oimmatidium
        std::vector<Ommatidium>* ommatidia = nullptr;
        // If sum is 0, then we have a special case for rendering the eye as we have no focal point
        // offsets specified for this eye (and hence the radius of the ommatidium is not known)
        float focal_point_sum = 0.0f;
        // Hard-coded number of faces making up an ommatidial element (the higher this is, the more round it will look)
        static constexpr int tube_faces = 18;
        // Rendering as cone. This is the number of vertices per cone.
        static constexpr int cone_vertices = tube_faces * 3 + 2;
        static constexpr int disc_vertices = tube_faces * 4 + 2;
        // Setter for cone_length must reinit vertices
        void set_cone_length (float _cone_length) { this->cone_length = _cone_length; this->reinit(); }
        float get_cone_length() { return this->cone_length; }
        // Setter for the disc width. To replace cone length? Or operate as alternative?
        void set_disc_width (float _disc_width) { this->disc_width = _disc_width; this->reinit(); }
        float get_disc_width() { return this->disc_width; }
    private:
        // User-modifiable ommatidial cone length which is used if there's no focal point offset
        float cone_length = 0.1f;
        // User-modifiable ommatidial disc width. If negative ignored?
        float disc_width = -1.0f;
    };

} // namespace comray
