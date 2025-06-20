#include "GenericCamera.h"
#include "DataRecordCamera.h"

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader

#include "PanoramicCameraDataTypes.h"

class PanoramicCamera : public DataRecordCamera<PanoramicCameraData>
{
public:
    PanoramicCamera(const std::string name);
    ~PanoramicCamera();
    void setStartRadius(float d);
    const char* getEntryFunctionName() const { return "__raygen__panoramic"; }
};
