/*!
 * @file 		UVJpegCompress.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVJpegCompress.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_compress.h"
#include "video_compress/jpeg.h"
#include "video_frame.h"
}
namespace yuri {
namespace uv_jpeg_compress {


IOTHREAD_GENERATOR(UVJpegCompress)

core::Parameters UVJpegCompress::configure()
{
	core::Parameters p = ultragrid::UVVideoCompress::configure();
	p.set_description("UVJpegCompress");
	return p;
}


UVJpegCompress::UVJpegCompress(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent,"uv_jpeg_compress", UV_COMPRESS_DETAIL(jpeg_info))
{
	IOTHREAD_INIT(parameters)
	if(!init_compressor("")) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create JPEG encoder");
	}
}

UVJpegCompress::~UVJpegCompress() noexcept
{
}


bool UVJpegCompress::set_param(const core::Parameter& param)
{
	return ultragrid::UVVideoCompress::set_param(param);
}

} /* namespace uv_jpeg_compress */
} /* namespace yuri */
