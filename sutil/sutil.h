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

#include <cuda_runtime.h>

#include <cstdlib>
#include <chrono>
#include <string>

#include <vector_types.h>

#include "sutilapi.h"

// Some helper macros to stringify the sample's name that comes in as a define
#define OPTIX_SAMPLE_NAME_STRINGIFY2(name) #name
#define OPTIX_SAMPLE_NAME_STRINGIFY(name) OPTIX_SAMPLE_NAME_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME OPTIX_SAMPLE_NAME_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)

namespace sutil
{

    static constexpr bool debug_sutil = false;

    enum BufferImageFormat
    {
        UNSIGNED_BYTE4,
        FLOAT4,
        FLOAT3
    };

    struct ImageBuffer
    {
        void* data =      nullptr;
        unsigned int      width = 0;
        unsigned int      height = 0;
        BufferImageFormat pixel_format;
    };

    // Return a path to a sample data file, or NULL if the file cannot be located.  The pointer
    // returned may point to a static array.
    SUTILAPI const char* sampleDataFilePath( const char* relativeFilePath );

    SUTILAPI size_t pixelFormatSize( BufferImageFormat format );

    // Create a cudaTextureObject_t for the given image file.  If the filename is empty or if
    // loading the file fails, return 1x1 texture with default color.
    SUTILAPI cudaTextureObject_t loadTexture( const std::string& filename, float3 default_color, cudaTextureDesc* tex_desc = nullptr );

    // Why is this not visible?
    SUTILAPI void displayBufferFile (const char* filename, const ImageBuffer& buffer, bool disable_srgb);

    // Blocking sleep call
    // seconds: Number of seconds to sleep
    SUTILAPI void sleep (int seconds );

    // Parse the string of the form <width>x<height> and return numeric values.
    SUTILAPI void parseDimensions(
        const char* arg,                    // String of form <width>x<height>
        int& width,                         // [out] width
        int& height );                      // [in]  height


    SUTILAPI void calculateCameraVariables(
        float3 eye,
        float3 lookat,
        float3 up,
        float  fov,
        float  aspect_ratio,
        float3& U,
        float3& V,
        float3& W,
        bool fov_is_vertical );

    // Get PTX, either pre-compiled with NVCC or JIT compiled by NVRTC.
    SUTILAPI const char* getPtxString(
        const char* sample,                 // Name of the sample, used to locate the input
                                            // file. NULL = only search the common /cuda dir
        const char* filename,               // Cuda C input file name
        const char** log = NULL );          // (Optional) pointer to compiler log string. If *log ==
                                            // NULL there is no output. Only valid until the next
                                            // getPtxString call

    // Ensures that width and height have the minimum size to prevent launch errors.
    SUTILAPI void ensureMinimumSize(
        int& width,                             // Will be assigned the minimum suitable width if too small.
        int& height);                           // Will be assigned the minimum suitable height if too small.

    // Ensures that width and height have the minimum size to prevent launch errors.
    SUTILAPI void ensureMinimumSize(
        unsigned& width,                        // Will be assigned the minimum suitable width if too small.
        unsigned& height);                      // Will be assigned the minimum suitable height if too small.

    SUTILAPI void reportErrorMessage( const char* message );

} // end namespace sutil
