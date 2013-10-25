/*!
 * @file 		UVJpegCompress.cpp
 * @author 		<Your name>
 * @date		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
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

MODULE_REGISTRATION_BEGIN("uv_jpeg_compress")
		REGISTER_IOTHREAD("uv_jpeg_compress",UVJpegCompress)
MODULE_REGISTRATION_END()

core::Parameters UVJpegCompress::configure()
{
	core::Parameters p = ultragrid::UVVideoCompress::configure();
	p.set_description("UVJpegCompress");
	return p;
}


UVJpegCompress::UVJpegCompress(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent,"uv_jpeg_compress", UV_COMPRESS_DETAIL(jpeg))
{
	IOTHREAD_INIT(parameters)
	if(!init_compressor("")) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create JPEG encoder");
	}
}

UVJpegCompress::~UVJpegCompress()
{
}


bool UVJpegCompress::set_param(const core::Parameter& param)
{
	return ultragrid::UVVideoCompress::set_param(param);
}

} /* namespace uv_jpeg_compress */
} /* namespace yuri */
