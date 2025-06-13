#pragma once

#include "GenericCameraDataTypes.h"
#include "GenericCamera.h"
#include <sutil/Quaternion.h>
#include <sutil/Matrix.h>

template<typename T>
class DataRecordCamera : public GenericCamera {
public:

    // Compile time choice of debug output in camera code
    static constexpr bool debug_cameras = false;
    static constexpr bool debug_memory = false;

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

    // Transform local vector into the world frame.
    // Seb thinks this should be named transformToWorld
    const float3 transformToLocal(const float3& vector) const
    {
        return (vector.x*ls.xAxis + vector.y*ls.yAxis + vector.z*ls.zAxis);
    }

    // Rotate camera around an axis in its own, camera space.
    void rotateLocallyAround(const float angle, const float3& localAxis)
    {
        std::cout << "rotateLocallyAround axis ("
                  << localAxis.x << "," << localAxis.y << "," << localAxis.z << ") by " << angle << " rad" << std::endl;
        // Project the axis and then perform the rotation of local axis around an axis
        // specified in the world frame transformToLocal should be named
        // transformToWorld.
        rotateAround(angle, transformToLocal(localAxis));
    }
    void rotateAround(const float angle, const float3& axis)
    {
        std::cout << "rotateAround axis (" << axis.x << "," << axis.y << "," << axis.z << ") by " << angle << " rad" << std::endl;
        // axis IS the world axis now, Tick.

        // Just performing an axis-angle rotation of the local space: A lot nicer.
        std::cout << "Rotate xAxis point (" << ls.xAxis.x << "," << ls.xAxis.y << "," << ls.xAxis.z << ") to ";
        ls.xAxis = rotatePoint (ls.xAxis, angle, axis);
        std::cout << ls.xAxis.x << "," << ls.xAxis.y << "," << ls.xAxis.z << ")\n";

        std::cout << "Rotate yAxis point (" << ls.yAxis.x << "," << ls.yAxis.y << "," << ls.yAxis.z << ") to ";
        ls.yAxis = rotatePoint (ls.yAxis, angle, axis);
        std::cout << ls.yAxis.x << "," << ls.yAxis.y << "," << ls.yAxis.z << ")\n";

        std::cout << "Rotate zAxis point (" << ls.zAxis.x << "," << ls.zAxis.y << "," << ls.zAxis.z << ") to ";
        ls.zAxis = rotatePoint (ls.zAxis, angle, axis);
        std::cout << ls.zAxis.x << "," << ls.zAxis.y << "," << ls.zAxis.z << ")\n";

        // Seems that ls has rotated as I expected it to. Tick.

        // What's up in eye3d?
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
            return true;
        } // else: you'd get a lot of noise cout-ing the else clause
        return false;
    }

    void forcePackAndCopyRecord(OptixProgramGroup& programGroup)
    {
        if constexpr (debug_cameras == true) {
            std::cout<< "Copying device memory for camera '"<<getCameraName()<<"'."<<std::endl;
        }

        // ProgramGroup contains the opaque type OptixModule along with a function pointer.
        OPTIX_CHECK (optixSbtRecordPackHeader (programGroup, reinterpret_cast<void*>(&this->sbtRecord)));
        CUDA_CHECK (cudaMemcpy (reinterpret_cast<void*>(d_record), &sbtRecord, sizeof(this->sbtRecord), cudaMemcpyHostToDevice));
        previous_sbtRecordData = this->sbtRecord.data;
    }

    virtual const CUdeviceptr& getRecordPtr() const {return d_record;}

    virtual float3* getRecordFrame()
    {
        std::cout << "DataRecordCamera does not implement getRecordFrame()\n";
        return nullptr;
    }

    void getLocalSpace (float3& xAxis, float3& yAxis, float3& zAxis)
    {
        xAxis = sbtRecord.data.localSpace.xAxis;
        yAxis = sbtRecord.data.localSpace.yAxis;
        zAxis = sbtRecord.data.localSpace.zAxis;
    }

protected:
    RaygenRecord<RaygenPosedContainer<T>> sbtRecord; // The sbtRecord associated with this camera
    T& specializedData = sbtRecord.data.specializedData; // Convenience reference
    LocalSpace& ls = sbtRecord.data.localSpace; // Convenience reference

private:
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
