#include "GenericCameraDataTypes.h"

struct CompoundEyeData
{
  size_t ommatidialCount;            // The number of ommatidia in this eye
  CUdeviceptr d_ommatidialArray = 0; // Points to a list of Ommatidium objects in VRAM
  uint32_t samplesPerOmmatidium = 1; // The number of samples taken from each ommatidium for this eye
  CUdeviceptr d_randomStates = 0;    // Pointer to this compound eye's random state buffer
  CUdeviceptr d_compoundBuffer = 0;  // Pointer to this compound eye's compound buffer, where samples from each ommatidium are stored

  inline bool operator==(const CompoundEyeData& other)
  {
    return this->ommatidialCount == other.ommatidialCount && this->d_ommatidialArray == other.d_ommatidialArray &&
           this->samplesPerOmmatidium == other.samplesPerOmmatidium && this->d_randomStates == other.d_randomStates &&
           this->d_compoundBuffer == other.d_compoundBuffer;
  }
};

// The ommatidium object
struct Ommatidium
{
  float3 relativePosition;
  float3 relativeDirection;
  float acceptanceAngleRadians;
  float focalPointOffset;
};

typedef RaygenPosedContainer<CompoundEyeData> CompoundEyePosedData;
typedef RaygenRecord<CompoundEyePosedData> CompoundEyePosedDataRecord;

// A simple record type that stores a pointer to another on-device record, used within the compound rendering pipeline to retrieve information from the projection pipeline
struct RecordPointer
{
  CUdeviceptr d_record = 0; // Points to another record on VRAM
};
typedef RaygenRecord<RecordPointer> RecordPointerRecord;