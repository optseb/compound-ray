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

        // Hard-coded number of faces making up an ommatidial 'flared tube'
        static constexpr int tube_faces = 18;
        // VisualModel::computeTube makes this many vertices per tube:
        static constexpr int tube_vertices = tube_faces * 4 + 2;
        static constexpr int cap_vertices = tube_faces + 1;
        static constexpr int mid_vertices = tube_faces * 2;
        // Hardcoded ommatidial flared tube start radius
        static constexpr float tube_radius = 0.005f;
        // Hardcoded ommatidial flared tube length
        static constexpr float tube_length = 0.04f;
        // Side colour of ommatidial tubes
        static constexpr std::array<float, 3> side_colour = { 0.7f, 0.7f, 0.7f };

        void updateColours()
        {
            if (ommData == nullptr) { return; }
            if (ommData->empty()) { return; }
            size_t n_verts = this->vertexColors.size(); // should be tube_vertices * n_omm
            if (n_verts == 0u) { return; } // model doesn't exist yet

            this->vertexColors.clear(); // Could re-write not clear/push
            size_t n_omm = ommData->size();

            // 3 colours, n_omm tubes, tube_vertices vertices per tube.
            if (n_verts != 3u * n_omm * static_cast<unsigned int>(tube_vertices)) {
                throw std::runtime_error ("CompoundEyeVisual: n_verts/n_omm sizes mismatch!");
            }

            for (size_t i = 0u; i < n_omm; ++i) {
                // Update the 3 RGB values in vertexColors tube_vertices times
                int j = 0;
                if (simple_flared_tubes) {
                    for (; j < tube_vertices; ++j) {
                        this->vertex_push ((*ommData)[i], this->vertexColors);
                    }
                } else {
                    for (; j < cap_vertices; ++j) {
                        this->vertex_push ((*ommData)[i], this->vertexColors);
                    }
                    for (; j < (cap_vertices + mid_vertices); ++j) {
                        this->vertex_push (side_colour, this->vertexColors);
                    }
                    for (; j < tube_vertices; ++j) {
                        this->vertex_push ((*ommData)[i], this->vertexColors);
                    }
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
            for (size_t i = 0u; i < n_omm; ++i) {
                // Ommatidia position/shape
                float3 rpos = (*ommatidia)[i].relativePosition;
                float3 rdir = (*ommatidia)[i].relativeDirection;
                float flare = (*ommatidia)[i].acceptanceAngleRadians;
                float foc_offset = (*ommatidia)[i].focalPointOffset;
                morph::vec<float, 3> start_coord = { rpos.x, rpos.y, rpos.z };
                morph::vec<float, 3> dir = { rdir.x, rdir.y, rdir.z };
                dir.renormalize();
                // focal point offset very likely not to be used correctly here:
                morph::vec<float, 3> end_coord = start_coord + dir * (foc_offset == 0.0f ? tube_length : foc_offset);
                // Colour comes from ommData
                std::array<float, 3> colour = (*ommData)[i];
                if (simple_flared_tubes) {
                    this->computeFlaredTube (this->idx, start_coord, end_coord, colour, colour, tube_radius, tube_faces, flare);
                } else {
                    this->computeOmmatidialTube (this->idx, start_coord, end_coord, colour, colour, tube_radius, tube_faces, flare);
                }
            }
        }

        //! compute r_end then call the main overload of computeOmmatidialTube
        void computeOmmatidialTube (GLuint& idx, morph::vec<float> start, morph::vec<float> end,
                                    std::array<float, 3> colStart, std::array<float, 3> colEnd,
                                    float r = 1.0f, int segments = 12, float flare = 0.0f)
        {
            // Find the length of the tube
            morph::vec<float> v = end - start;
            float l = v.length();
            // Compute end radius from the length and the flare angle:
            float r_add = l * std::tan (std::abs(flare)) * (flare > 0.0f ? 1.0f : -1.0f);
            float r_end = r + r_add;
            // Now call into the other overload:
            this->computeOmmatidialTube (idx, start, end, colStart, colEnd, r, r_end, segments);
        }

        /*!
         * Create a special flared tube from \a start to \a end, with radius \a r at the start and a colour
         * which transitions from the colour \a colStart to \a colEnd. The radius of the end is
         * r_end, given as a function argument. The *side* of the tubes is hard coded (so this is app-specific)
         *
         * \param idx The index into the 'vertex array'
         * \param start The start of the tube
         * \param end The end of the tube
         * \param colStart The tube starting colour
         * \param colEnd The tube's ending colour
         * \param r Radius of the tube's start cap
         * \param r_end radius of the end cap
         * \param segments Number of segments used to render the tube
         */
        void computeOmmatidialTube (GLuint& idx, morph::vec<float> start, morph::vec<float> end,
                                    std::array<float, 3> colStart, std::array<float, 3> colEnd,
                                    float r = 1.0f, float r_end = 1.0f, int segments = 12)
        {
            // The vector from start to end defines a vector and a plane. Find a
            // 'circle' of points in that plane.
            morph::vec<float> vstart = start;
            morph::vec<float> vend = end;
            morph::vec<float> v = vend - vstart;
            v.renormalize();

            // circle in a plane defined by a point (v0 = vstart or vend) and a normal
            // (v) can be found: Choose random vector vr. A vector inplane = vr ^ v. The
            // unit in-plane vector is inplane.normalise. Can now use that vector in the
            // plan to define a point on the circle. Note that this starting point on
            // the circle is at a random position, which means that this version of
            // computeTube is useful for tubes that have quite a few segments.
            morph::vec<float> rand_vec;
            rand_vec.randomize();
            morph::vec<float> inplane = rand_vec.cross(v);
            inplane.renormalize();

            // Now use parameterization of circle inplane = p1-x1 and
            // c1(t) = ( (p1-x1).normalized sin(t) + v.normalized cross (p1-x1).normalized * cos(t) )
            // c1(t) = ( inplane sin(t) + v * inplane * cos(t)
            morph::vec<float> v_x_inplane = v.cross(inplane);

            // Push the central point of the start cap - this is at location vstart
            this->vertex_push (vstart, this->vertexPositions);
            this->vertex_push (-v, this->vertexNormals);
            this->vertex_push (colStart, this->vertexColors);

            // Start cap vertices. Draw as a triangle fan, but record indices so that we
            // only need a single call to glDrawElements.
            for (int j = 0; j < segments; j++) {
                // t is the angle of the segment
                float t = j * morph::mathconst<float>::two_pi/(float)segments;
                morph::vec<float> c = inplane * std::sin(t) * r + v_x_inplane * std::cos(t) * r;
                this->vertex_push (vstart+c, this->vertexPositions);
                this->vertex_push (-v, this->vertexNormals);
                this->vertex_push (colStart, this->vertexColors);
            }

            // Intermediate, near start cap. Normals point in direction c
            for (int j = 0; j < segments; j++) {
                float t = j * morph::mathconst<float>::two_pi/(float)segments;
                morph::vec<float> c = inplane * std::sin(t) * r + v_x_inplane * std::cos(t) * r;
                this->vertex_push (vstart+c, this->vertexPositions);
                c.renormalize();
                this->vertex_push (c, this->vertexNormals);
                this->vertex_push (side_colour, this->vertexColors);
            }

            // Intermediate, near end cap. Normals point in direction c
            for (int j = 0; j < segments; j++) {
                float t = (float)j * morph::mathconst<float>::two_pi/(float)segments;
                morph::vec<float> c = inplane * std::sin(t) * r_end + v_x_inplane * std::cos(t) * r_end;
                this->vertex_push (vend+c, this->vertexPositions);
                c.renormalize();
                this->vertex_push (c, this->vertexNormals);
                this->vertex_push (side_colour, this->vertexColors);
            }

            // Bottom cap vertices
            for (int j = 0; j < segments; j++) {
                float t = (float)j * morph::mathconst<float>::two_pi/(float)segments;
                morph::vec<float> c = inplane * std::sin(t) * r_end + v_x_inplane * std::cos(t) * r_end;
                this->vertex_push (vend+c, this->vertexPositions);
                this->vertex_push (v, this->vertexNormals);
                this->vertex_push (colEnd, this->vertexColors);
            }

            // Bottom cap. Push centre vertex as the last vertex.
            this->vertex_push (vend, this->vertexPositions);
            this->vertex_push (v, this->vertexNormals);
            this->vertex_push (colEnd, this->vertexColors);

            // Note: number of vertices = segments * 4 + 2.
            int nverts = (segments * 4) + 2;

            // After creating vertices, push all the indices.
            GLuint capMiddle = idx;
            GLuint capStartIdx = idx + 1u;
            GLuint endMiddle = idx + (GLuint)nverts - 1u;
            GLuint endStartIdx = capStartIdx + (3u * segments);

            // Start cap
            for (int j = 0; j < segments-1; j++) {
                this->indices.push_back (capMiddle);
                this->indices.push_back (capStartIdx + j);
                this->indices.push_back (capStartIdx + 1 + j);
            }
            // Last one
            this->indices.push_back (capMiddle);
            this->indices.push_back (capStartIdx + segments - 1);
            this->indices.push_back (capStartIdx);

            // Middle sections
            for (int lsection = 0; lsection < 3; ++lsection) {
                capStartIdx = idx + 1 + lsection*segments;
                endStartIdx = capStartIdx + segments;
                // This does sides between start and end. I want to do this three times.
                for (int j = 0; j < segments; j++) {
                    // Triangle 1
                    this->indices.push_back (capStartIdx + j);
                    if (j == (segments-1)) {
                        this->indices.push_back (capStartIdx);
                    } else {
                        this->indices.push_back (capStartIdx + 1 + j);
                    }
                    this->indices.push_back (endStartIdx + j);
                    // Triangle 2
                    this->indices.push_back (endStartIdx + j);
                    if (j == (segments-1)) {
                        this->indices.push_back (endStartIdx);
                    } else {
                        this->indices.push_back (endStartIdx + 1 + j);
                    }
                    if (j == (segments-1)) {
                        this->indices.push_back (capStartIdx);
                    } else {
                        this->indices.push_back (capStartIdx + j + 1);
                    }
                }
            }

            // Bottom cap
            for (int j = 0; j < segments-1; j++) {
                this->indices.push_back (endMiddle);
                this->indices.push_back (endStartIdx + j);
                this->indices.push_back (endStartIdx + 1 + j);
            }
            // Last one
            this->indices.push_back (endMiddle);
            this->indices.push_back (endStartIdx + segments - 1);
            this->indices.push_back (endStartIdx);

            // Update idx
            idx += nverts;
        }

        void toggle_simple_flared()
        {
            this->simple_flared_tubes = !this->simple_flared_tubes;
            this->reinit();
        }

        bool get_simple_flared() { return this->simple_flared_tubes; }

        // The colours detected by each ommatidium
        std::vector<std::array<float, 3>>* ommData = nullptr;
        // The position and orientation of each oimmatidium
        std::vector<Ommatidium>* ommatidia = nullptr;
    private:
        // Draw simple flares or our tweaked ommatidial tubes? (runtime modifiable)
        bool simple_flared_tubes = false;
    };

} // namespace comray
