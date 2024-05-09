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
        // Hardcoded ommatidial flared tube start radius
        static constexpr float tube_radius = 0.005f;
        // Hardcoded ommatidial flared tube length
        static constexpr float tube_length = 0.04f;

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
                for (int j = 0; j < tube_vertices; ++j) {
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
                this->computeFlaredTube (this->idx, start_coord, end_coord, colour, colour, tube_radius, tube_faces, flare);
            }
        }

        // The colours detected by each ommatidium
        std::vector<std::array<float, 3>>* ommData = nullptr;
        // The position and orientation of each oimmatidium
        std::vector<Ommatidium>* ommatidia = nullptr;
    };

} // namespace comray
