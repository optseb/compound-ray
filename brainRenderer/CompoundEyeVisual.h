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

        //! Initialize vertex buffer objects and vertex array object.
        void initializeVertices()
        {
            this->vertexPositions.clear();
            this->vertexNormals.clear();
            this->vertexColors.clear();
            this->indices.clear();

            if (ommData == nullptr || ommatidia == nullptr) { return; }
            if (ommatidia != nullptr && ommatidia->empty()) { return; }
            if (ommData != nullptr && ommData->empty()) { return; }

            if (ommData->size() != ommatidia->size()) {
                throw std::runtime_error ("sizes mismatch!");
            }

            // Draw ommatidia
            size_t n_omm = ommData->size();
            for (size_t i = 0u; i < n_omm; ++i) {
                // at location float3 ommatidia[i].relativePosition,
                // draw cone with orientation float3 ommatidia[i].relativeDirection
                // and angle float ommatidia[i].acceptanceAngleRadians.
                // Somehow use float ommatidia[i].focalPointOffset
                // Colour is ommData[i]
                float3 rpos = (*ommatidia)[i].relativePosition;
                float3 rdir = (*ommatidia)[i].relativeDirection;
                morph::vec<float, 3> start_coord = { rpos.x, rpos.y, rpos.z };
                morph::vec<float, 3> dir = { rdir.x, rdir.y, rdir.z };
                dir.renormalize();
                morph::vec<float, 3> end_coord = start_coord + dir * 0.01f; // just a hack
                float radius = 0.005f; // just a hack
                std::array<float, 3> colour = (*ommData)[i];
                this->computeTube (this->idx, start_coord, end_coord, colour, colour, radius, 18);
            }
        }

        // The colours detected by each ommatidium
        std::vector<std::array<float, 3>>* ommData = nullptr;
        // The position and orientation of each oimmatidium
        std::vector<Ommatidium>* ommatidia = nullptr;
    };

} // namespace comray
