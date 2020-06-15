// The virtual superclass of all camera objects

#pragma once

#define DEBUG

#include <optix.h>
#include <sutil/Quaternion.h>
#include <sutil/Exception.h>

#include <iostream>

#include "GenericCameraDataTypes.h"

class GenericCamera {
  public:
    //Constructor/Destructor
    GenericCamera(int progGroupID);
    GenericCamera();
    ~GenericCamera();

    const float3& getPosition() const { return position; }
    void setPosition(const float3 pos);
    // Returns the local frame of the camera (always unit vectors)
    void getLocalFrame(float3& x, float3& y, float3& z) const;

    // Allocates device memory for the SBT record
    virtual void allocateRecord() = 0;
    // Packs and then copies the data onto the device
    virtual void packAndCopyRecord(OptixProgramGroup& programGroup) = 0;
    // Gets a pointer to the data on the device.
    const CUdeviceptr& getRecordPtr() const;
    const int getProgramGroupID() const { return programGroupID; }

  protected:
    // The below allow access to device-side control objects
    CUdeviceptr d_record = 0;// Stores the SBT record required by this camera
    //const OptixProgramGroup& programGroup;// Stores a reference to the associated program group

  private:
    float3 position;
    //Quaternion orientation;
    const int programGroupID = 0; // Horrible hacky code that stores an ID for MulticamScene to reference later in order for it to assign the correct program group to this camera.
};