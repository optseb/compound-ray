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

#include <sutil/Preprocessor.h>

#include <stdint.h>

#ifndef __CUDA_ARCH__ // i.e. do the stdout only on the host
#include <iostream>
#endif

// A BufferView of things of type T, where T may be a native type like int, float, etc
// or a type like float3, float2 etc. T should not be some user-defined complex type. If
// T is float3, then the 'elements' are float if T is float then the elements are float
// also.
template <typename T>
struct BufferView
{
    CUdeviceptr  data            CONST_STATIC_INIT( 0 );
    uint32_t     elmt_count      CONST_STATIC_INIT( 0 ); // This, confusingly previously
                                                         // named 'count', is
                                                         // 'ELEMENT-of-object' count
                                                         // rather than object count.
                                                         // if T is float2, then it's
                                                         // the number of floats (not
                                                         // the number of float2s).
    uint16_t     byte_stride     CONST_STATIC_INIT( 0 );
    uint16_t     elmt_byte_size  CONST_STATIC_INIT( 0 ); // *not* necessarily the same as sizeof(T)

    SUTIL_HOSTDEVICE bool isValid() const
    { return static_cast<bool>( data ); }

    SUTIL_HOSTDEVICE operator bool() const
    { return isValid(); }

    SUTIL_HOSTDEVICE const T& operator[]( uint32_t idx ) const
    {
#ifndef __CUDA_ARCH__ // i.e. do the stdout only on the host
        std::cerr << "BufferView["<<idx<<"] is ptr " << reinterpret_cast<void*>(data)
                  << " + idx * (byte_stride ? byte_stride : sizeof( T ) ) = "
                  << (idx * (byte_stride ? byte_stride : sizeof(T)))
                  << " and elmt_byte_size = " << elmt_byte_size << "/sizeof(T): " << sizeof(T) << std::endl;
#endif
        return *reinterpret_cast<T*>(data + idx * actual_stride());
    }

    // The stride is either the byte_stride if defined (larger than sizeof(T)) or sizeof(T)
    SUTIL_HOSTDEVICE size_t actual_stride() const { return (byte_stride ? byte_stride : sizeof(T)); }
    // Return the buffers data size in bytes
    SUTIL_HOSTDEVICE uint32_t size_bytes() const { return elmt_count * elmt_byte_size; }
    // Return the count - the object_count of things of type T
    SUTIL_HOSTDEVICE uint32_t count() const { return size_bytes() / sizeof(T); }
    // The element size in bytes is actually just a member
    SUTIL_HOSTDEVICE uint32_t element_size_bytes() const { return elmt_byte_size; }
    // How many elements in an object?
    SUTIL_HOSTDEVICE uint32_t object_size_elmts() const { return sizeof(T) / elmt_byte_size; }
    // Return an object size in bytes. simply sizeof(T)
    SUTIL_HOSTDEVICE uint32_t object_size_bytes() const { return sizeof(T); }
    // How many things of type T are there?
    SUTIL_HOSTDEVICE uint32_t object_count() const { return size_bytes() / sizeof(T); }
};

#ifndef __CUDA_ARCH__ // i.e. do the stdout only on the host
#include <vector>
#include <cuda_runtime.h>

namespace cuda
{
    // small helper to copy a BufferView to local CPU ram, most likely for debugging
    template <typename T>
    struct CopiedBufferView
    {
        BufferView<T> bv; // The CUDA buffer view of data

        std::vector<T> bv_data = {};

        CopiedBufferView (const BufferView<T>& _bv)
        {
            // For simplicity don't handle funky byte strides
            if (bv.byte_stride > 0) {
                std::cerr << "Don't handle byte_stride > 0" << std::endl;
                return;
            }

            this->bv = _bv;
            this->bv_data.resize (bv.elmt_count);
            void* dst = bv_data.data();

            size_t nbytes = bv.elmt_count * bv.elmt_byte_size;

            //std::cout << "copy " << nbytes << " bytes from " << reinterpret_cast<void*>(bv.data)
            //          << " to " << dst << " elmt_byte_size: " << bv.elmt_byte_size
            //          << " and sizeof(T): " << sizeof(T) << std::endl;
            auto mc_rtn = cudaMemcpy (dst, reinterpret_cast<void*>(bv.data), nbytes,  cudaMemcpyDeviceToHost);
            if (mc_rtn) { std::cout << "cudaMemcpy returned error: " << mc_rtn << std::endl; }
        }

        const T operator[]( uint32_t idx ) const
        {
            if (idx < bv_data.size()) {
                return bv_data[idx];
            } else { return T{}; }
        }
    };
}
#endif
