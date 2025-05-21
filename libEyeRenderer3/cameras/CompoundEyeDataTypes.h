#pragma once
#include "GenericCameraDataTypes.h"

struct CompoundEyeData // 52 bytes or 64 with padding
{
  size_t ommatidialCount = 0;           // 8 The number of ommatidia in this eye
  CUdeviceptr d_ommatidialArray = 0;    // 8 Points to a list of Ommatidium objects in VRAM
  uint32_t samplesPerOmmatidium = 1;    // 4 The number of samples taken from each ommatidium for this eye
  CUdeviceptr d_randomStates = 0;       // 8 Pointer to this compound eye's random state buffer
  CUdeviceptr d_compoundBuffer = 0;     // 8 Pointer to this compound eye's compound buffer, where samples from each ommatidium are stored
  CUdeviceptr d_compoundAvgBuffer = 0;  // 8 Pointer to a buffer to contain the average of all the samples for each ommatidium
  bool randomsConfigured = false;       // 8 Flag to track whether the random state buffer has been configured, as they need to be before being used. Ultimately this will be replaced by explicitly configuring randoms on memory set. TODO(RANDOMS)
  size_t pad1 = 0;
  uint32_t pad2 = 0;

  inline bool operator==(const CompoundEyeData& other)
  {
    return this->ommatidialCount == other.ommatidialCount && this->d_ommatidialArray == other.d_ommatidialArray &&
           this->samplesPerOmmatidium == other.samplesPerOmmatidium && this->d_randomStates == other.d_randomStates &&
           this->d_compoundBuffer == other.d_compoundBuffer && this->d_compoundAvgBuffer == other.d_compoundAvgBuffer &&
           this->randomsConfigured == other.randomsConfigured;//TODO(RANDOMS): Remove the 'randomsConfigured' check.
  }
};

// The ommatidium object. 32 bytes.
struct Ommatidium
{
  float3 relativePosition;        // 12
  float3 relativeDirection;       // 12
  float acceptanceAngleRadians;   // 4
  float focalPointOffset;         // 4
};

// A simple record type that stores a pointer to another on-device record, used within the compound
// rendering pipeline to retrieve information from the projection pipeline
// 8 bytes or 16 if I pad it.
struct RecordPointer
{
  CUdeviceptr d_record = 0; // Points to another record on VRAM
  CUdeviceptr pad = 0;      // neither harms nor heals on its own
};
