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


#include <sampleConfig.h>
#include <sutil/Exception.h>
#include <sutil/PPMLoader.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#if !defined( _WIN32 )
#include <dirent.h>
#endif

namespace sutil
{

static void SavePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw std::invalid_argument( "Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw std::invalid_argument( "Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw std::runtime_error( "Could not open file for SavePPM" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) ) << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid * hgt * chan * ( is_float ? 4 : 1 ) );

    OutFile.close();
}


static bool dirExists( const char* path )
{
#if defined( _WIN32 )
    DWORD attrib = GetFileAttributes( path );
    return ( attrib != INVALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return false;

    closedir( dir );
    return true;
#endif
}

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static std::string existingFilePath( const char* directory, const char* relativeSubDir, const char* relativePath )
{
    std::string path = directory ? directory : "";
    path += '/';
    path += relativeSubDir;
    path += '/';
    path += relativePath;
    return fileExists( path ) ? path : "";
}

std::string getSampleDir()
{
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( dirExists( directory ) )
            return directory;
    }

    throw Exception( "sutil::getSampleDir couldn't locate an existing sample directory" );
}

/**
 * Also now allows for direct paths.
 */
const char* sampleDataFilePath( const char* relativePath )
{
    if(relativePath[0] == '/' && fileExists(relativePath))
      return relativePath;

    static std::string s;

    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory )
        {
            s = existingFilePath( directory, "data", relativePath );
            if( !s.empty() )
            {
                return s.c_str();
            }
        }
    }
    throw Exception( ( std::string{"sutil::sampleDataFilePath couldn't locate "} + relativePath ).c_str() );
}


size_t pixelFormatSize( BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            throw Exception( "sutil::pixelFormatSize: Unrecognized buffer format" );
    }
}


cudaTextureObject_t loadTexture( const std::string& filename, float3 default_color, cudaTextureDesc* tex_desc )
{
    bool   isHDR = false;
    size_t len   = filename.length();
    if( len >= 3 )
    {
        isHDR = ( filename[len - 3] == 'H' || filename[len - 3] == 'h' ) &&
		( filename[len - 2] == 'D' || filename[len - 2] == 'd' ) &&
		( filename[len - 1] == 'R' || filename[len - 1] == 'r' );
    }
    if( isHDR )
    {
        std::cerr << "HDR texture loading not yet implemented" << std::endl;
        return 0;
    }
    else
    {
        return loadPPMTexture( filename, default_color, tex_desc );
    }
}

void displayBufferFile( const char* filename, const ImageBuffer& buffer, bool disable_srgb_conversion )
{
    // TODO: use stb_image_write to output PNG
    unsigned int width = 0;
    unsigned int height = 0;

    void* imageData = buffer.data;
    width           = static_cast<unsigned int>( buffer.width );
    height          = static_cast<unsigned int>( buffer.height );

    std::vector<unsigned char> pix( width * height * 3 );

    BufferImageFormat buffer_format = buffer.pixel_format;

    const float gamma_inv = 1.0f / 2.2f;

    switch( buffer_format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            // Data is upside down
            for( int j = height - 1; j >= 0; --j )
            {
                unsigned char* dst = &pix[0] + ( 3 * width * ( height - 1 - j ) );
                unsigned char* src = ( (unsigned char*)imageData ) + ( 4 * width * j );
                for( int i = 0; i < width; i++ )
                {
                    *dst++ = *( src + 0 );
                    *dst++ = *( src + 1 );
                    *dst++ = *( src + 2 );
                    src += 4;
                }
            }
            break;

        case BufferImageFormat::FLOAT3:
            // This buffer is upside down
            for( int j = height - 1; j >= 0; --j )
            {
                unsigned char* dst = &pix[0] + ( 3 * width * ( height - 1 - j ) );
                float*         src = ( (float*)imageData ) + ( 3 * width * j );
                for( int i = 0; i < width; i++ )
                {
                    for( int elem = 0; elem < 3; ++elem )
                    {
                        int P;
                        if( disable_srgb_conversion )
                            P = static_cast<int>( ( *src++ ) * 255.0f );
                        else
                            P = static_cast<int>( std::pow( *src++, gamma_inv ) * 255.0f );
                        unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
                        *dst++               = static_cast<unsigned char>( Clamped );
                    }
                }
            }
            break;

        case BufferImageFormat::FLOAT4:
            // This buffer is upside down
            for( int j = height - 1; j >= 0; --j )
            {
                unsigned char* dst = &pix[0] + ( 3 * width * ( height - 1 - j ) );
                float*         src = ( (float*)imageData ) + ( 4 * width * j );
                for( int i = 0; i < width; i++ )
                {
                    for( int elem = 0; elem < 3; ++elem )
                    {
                        int P;
                        if( disable_srgb_conversion )
                            P = static_cast<int>( ( *src++ ) * 255.0f );
                        else
                            P = static_cast<int>( std::pow( *src++, gamma_inv ) * 255.0f );
                        unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
                        *dst++               = static_cast<unsigned char>( Clamped );
                    }

                    // skip alpha
                    src++;
                }
            }
            break;

        default:
            fprintf( stderr, "Unrecognized buffer data type or format.\n" );
            exit( 2 );
            break;
    }

    SavePPM( &pix[0], filename, width, height, 3 );
}

void parseDimensions( const char* arg, int& width, int& height )
{
    // look for an 'x': <width>x<height>
    size_t width_end    = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char* height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }
    const std::string err = "Failed to parse width, height from string '" + std::string( arg ) + "'";
    throw std::invalid_argument( err.c_str() );
}


#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) ); \
    } while( 0 )

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str() );
    if( file.good() )
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

#if CUDA_NVRTC_ENABLED

static void getCuStringFromFile( std::string& cu, std::string& location, const char* sample_name, const char* filename )
{
    std::vector<std::string> source_locations;

    const std::string base_dir = getSampleDir();

    // Potential source locations (in priority order)
    if( sample_name )
        source_locations.push_back( base_dir + '/' + sample_name + '/' + filename );
    source_locations.push_back( base_dir + "/cuda/" + filename );

    for( const std::string& loc : source_locations )
    {
        // Try to get source code from file
        if( readSourceFile( cu, loc ) )
        {
            location = loc;
            return;
        }
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error( "Couldn't open source file " + std::string( filename ) );
}

static std::string g_nvrtcLog;

static void getPtxFromCuString( std::string& ptx, const char* sample_name, const char* cu_source, const char* name, const char** log_string )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = getSampleDir();

    // Set sample dir as the primary include path
    std::string sample_dir;
    if( sample_name )
    {
        sample_dir = std::string( "-I" ) + base_dir + '/' + sample_name;
        options.push_back( sample_dir.c_str() );
    }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {SAMPLES_ABSOLUTE_INCLUDE_DIRS};
    const char*              rel_dirs[] = {SAMPLES_RELATIVE_INCLUDE_DIRS};

    for( const char* dir : abs_dirs )
    {
        include_dirs.push_back( std::string( "-I" ) + dir );
    }
    for( const char* dir : rel_dirs )
    {
        include_dirs.push_back( "-I" + base_dir + '/' + dir );
    }
    for( const std::string& dir : include_dirs)
    {
        options.push_back( dir.c_str() );
    }

    // Collect NVRTC options
    const char*  compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &ptx_size ) );
    ptx.resize( ptx_size );
    NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &ptx[0] ) );

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}

#else  // CUDA_NVRTC_ENABLED

static std::string samplePTXFilePath( const char* sampleName, const char* fileName )
{
    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_PTX_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_PTX_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" ),
        SAMPLES_PTX_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory )
        {
            std::string path = directory;
            path += '/';
            path += sampleName ? sampleName : "cuda_compile_ptx";
            path += "_generated_";
            path += fileName;
            path += ".ptx";
            if( fileExists( path ) )
                return path;
        }
    }

    std::string error = "sutil::samplePTXFilePath couldn't locate ";
    error += fileName;
    error += " for sample ";
    error += sampleName;
    throw Exception( error.c_str() );
}

static void getPtxStringFromFile( std::string& ptx, const char* sample_name, const char* filename )
{
    const std::string sourceFilePath = samplePTXFilePath( sample_name, filename );

    // Try to open source PTX file
    if( !readSourceFile( ptx, sourceFilePath ) )
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error( err.c_str() );
    }
}

#endif  // CUDA_NVRTC_ENABLED

struct PtxSourceCache
{
    std::map<std::string, std::string*> map;
    ~PtxSourceCache()
    {
        for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;
    }
};
static PtxSourceCache g_ptxSourceCache;

const char* getPtxString( const char* sample, const char* filename, const char** log )
{
    if( log )
        *log = NULL;

    std::string *                                 ptx, cu;
    std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find( key );

    if( elem == g_ptxSourceCache.map.end() )
    {
        ptx = new std::string();
#if CUDA_NVRTC_ENABLED
        std::string location;
        getCuStringFromFile( cu, location, sample, filename );
        getPtxFromCuString( *ptx, sample, cu.c_str(), location.c_str(), log );
#else
        getPtxStringFromFile( *ptx, sample, filename );
#endif
        g_ptxSourceCache.map[key] = ptx;
    }
    else
    {
        ptx = elem->second;
    }

    return ptx->c_str();
}

void ensureMinimumSize( int& w, int& h )
{
    if( w <= 0 )
        w = 1;
    if( h <= 0 )
        h = 1;
}

void ensureMinimumSize( unsigned& w, unsigned& h )
{
    if( w == 0 )
        w = 1;
    if( h == 0 )
        h = 1;
}

void reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
#if defined( _WIN32 ) && defined( RELEASE_PUBLIC )
    {
        char s[2048];
        sprintf( s, "OptiX Error: %s", message );
        MessageBoxA( 0, s, "OptiX Error", MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL );
    }
#endif
}

} // namespace sutil
