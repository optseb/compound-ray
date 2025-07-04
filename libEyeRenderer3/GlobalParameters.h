//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <vector_types.h>
#include <stdint.h>

#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/Light.h>
#include <cuda/MaterialData.h>

namespace globalParameters
{

const uint32_t NUM_PAYLOAD_VALUES = 4u;


struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};


struct LaunchParams
{
    uchar4*                  frame_buffer; // An output buffer for non-compound eye cameras
    int32_t                  max_depth;

    CUdeviceptr              compoundBufferPtr;// Pointer to an on-device buffer for compound eye handling
    uint32_t                 compoundBufferWidth;
    uint32_t                 compoundBufferHeight;
    uint32_t                 compoundBufferDepth;
    uint32_t                 frame;// The current frame

    bool                     lighting;
    cuda::BufferView<Light::Point> lights;
    float3                   miss_color;
    OptixTraversableHandle   handle;
};


struct PayloadRadiance
{
    float3 result;
    float  importance;
    int    depth;
};


struct PayloadOcclusion
{
};


} // end namespace whitted
