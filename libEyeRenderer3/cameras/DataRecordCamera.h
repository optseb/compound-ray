#pragma once

#include "GenericCameraDataTypes.h"
#include "GenericCamera.h"
#include <sutil/Quaternion.h>
#include <sutil/Matrix.h>

template<typename T>
class DataRecordCamera : public GenericCamera {
public:

    // Compile time choice of debug output in camera code
    static constexpr bool debug_cameras = true;
    static constexpr bool debug_memory = true;

    DataRecordCamera(const std::string name) : GenericCamera(name)
    {
        // Allocate space for the record
        allocateRecord();
    }
    virtual ~DataRecordCamera()
    {
        // Free the allocated record
        freeRecord();
    }

    const float3& getPosition() const { return sbtRecord.data.position; }
    void setPosition(const float3 pos)
    {
        sbtRecord.data.position.x = pos.x;
        sbtRecord.data.position.y = pos.y;
        sbtRecord.data.position.z = pos.z;
    }

    void setLocalSpace(const float3 xAxis, const float3 yAxis, const float3 zAxis)
    {
        ls.xAxis = xAxis;
        ls.yAxis = yAxis;
        ls.zAxis = zAxis;
    }
    void lookAt(const float3& pos)
    {
        lookAt(pos, make_float3(0.0f, 1.0f, 0.0f));
    }
    void lookAt(const float3& pos, const float3& upVector)
    {
        ls.zAxis = normalize(pos - sbtRecord.data.position);
        ls.xAxis = normalize(cross(ls.zAxis, upVector));
        ls.yAxis = normalize(cross(ls.xAxis, ls.zAxis));
    }
    void resetPose()
    {
        ls.xAxis = {1.0f, 0.0f, 0.0f};
        ls.yAxis = {0.0f, 1.0f, 0.0f};
        ls.zAxis = {0.0f, 0.0f, 1.0f};
        sbtRecord.data.position = {0.0f, 0.0f, 0.0f};
    }


    const float3 transformToLocal(const float3& vector) const
    {
        return (vector.x*ls.xAxis + vector.y*ls.yAxis + vector.z*ls.zAxis);
    }
    void rotateLocallyAround(const float angle, const float3& localAxis)
    {
        // Project the axis and then perform the rotation
        rotateAround(angle, transformToLocal(localAxis));
    }
    void rotateAround(const float angle, const float3& axis)
    {
        // Just performing an axis-angle rotation of the local space: A lot nicer.
        ls.xAxis = rotatePoint(ls.xAxis, angle, axis);
        ls.yAxis = rotatePoint(ls.yAxis, angle, axis);
        ls.zAxis = rotatePoint(ls.zAxis, angle, axis);
    }

    void moveLocally(const float3& localStep)
    {
        move(transformToLocal(localStep));
    }
    void move(const float3& step)
    {
        sbtRecord.data.position += step;
    }

    float3 rotatePoint(const float3& point, const float angle, const float3& axis)
    {
        const float3 normedAxis = normalize(axis);
        return (cos(angle)*point + sin(angle)*cross(normedAxis, point) + (1 - cos(angle))*dot(normedAxis, point)*normedAxis);
    }

    bool packAndCopyRecordIfChanged(OptixProgramGroup& programGroup)
    {
        // Only copy the data across if it's changed
        if(previous_sbtRecordData != sbtRecord.data)
        {
            if constexpr (debug_cameras == true) {
                std::cout << "ALERT: The following copy was triggered as the sbt record was flagged as changed:" <<std::endl;
            }
            forcePackAndCopyRecord(programGroup);
            std::cout << std::endl;
            return true;
        } // else: you'd get a lot of noise cout-ing the else clause
        return false;
    }

    int mycounter = 0;

    void forcePackAndCopyRecord(OptixProgramGroup& programGroup)
    {
        // ProgramGroup contains the opaque type OptixModule along with a function pointer.
        OptixResult rph_res = optixSbtRecordPackHeader (programGroup, reinterpret_cast<void*>(&this->sbtRecord));
        std::cout << "optixSbtRecordPackHeader PACKED header into &sbtRecord " << &this->sbtRecord
                  << " with result: " << (int)rph_res <<  std::endl;
        OPTIX_CHECK (rph_res);

#if 0
        CUDA_CHECK (cudaMemset (reinterpret_cast<void*>(d_record), 0, sizeof(this->sbtRecord)));
#else
        if (mycounter < 149000000) {
            auto mc_res = cudaMemcpy (reinterpret_cast<void*>(d_record), &sbtRecord, sizeof(this->sbtRecord), cudaMemcpyHostToDevice);
            std::cout << "cudaMemcpy " << mycounter++ << " to device address " << d_record << " from &sbtRecord " << &this->sbtRecord
                      << " result: " << (int)mc_res << std::endl;
            CUDA_CHECK (mc_res);
        }
#endif
        previous_sbtRecordData = this->sbtRecord.data;
    }

    virtual const CUdeviceptr& getRecordPtr() const {return d_record;}

    virtual float3* getRecordFrame()
    {
        std::cout << "DataRecordCamera does not implement getRecordFrame()\n";
        return nullptr;
    }

protected:
    RaygenRecord<RaygenPosedContainer<T>> sbtRecord; // The sbtRecord associated with this camera
    T& specializedData = sbtRecord.data.specializedData; // Convenience reference
    LocalSpace& ls = sbtRecord.data.localSpace; // Convenience reference

private:
    static const LocalSpace BASE_LOCALSPACE;// A base localspace to use for rotations.

    CUdeviceptr d_record = 0;// Stores the pointer to the SBT record

    // Change tracking duplicates (done by keeping an old copy and comparing)
    RaygenPosedContainer<T> previous_sbtRecordData;

    void allocateRecord()
    {
        if constexpr (debug_memory == true) {
            std::cout << "Allocating camera SBT record on device (size: "<< sizeof(sbtRecord) << ")..." << std::endl;
        }
        if (d_record != 0) {
            if constexpr (debug_memory == true) {
                std::cout << "  WARN: Attempt to allocate camera SBT record was made when one is already allocated." << std::endl;
            }
            return;
        }
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_record ), sizeof(sbtRecord)) );

        if constexpr (debug_memory == true) {
            std::cout << "SBT allocated at d_record=" << d_record << " and d_record%16 = " << (d_record%16) << std::endl;
        }
    }
    void freeRecord()
    {
        if constexpr (debug_memory == true) { std::cout << "Freeing camera SBT record..." << std::endl; }
        if (d_record != 0) { CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_record)) ); }
    }
};
