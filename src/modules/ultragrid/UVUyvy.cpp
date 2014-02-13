/*!
 * @file 		UVUyvy.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		22.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVUyvy.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_compress.h"
#include "video_compress/uyvy.h"
}
#include "uv_video.h"
namespace yuri {
namespace uv_uyvy {


IOTHREAD_GENERATOR(UVUyvy)

core::Parameters UVUyvy::configure()
{
	core::Parameters p = ultragrid::UVVideoCompress::configure();
	p.set_description("UV UYVY");
	return p;
}


UVUyvy::UVUyvy(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent,"uv_uyvy", UV_COMPRESS_DETAIL(uyvy))
{
	IOTHREAD_INIT(parameters)
	if(!init_compressor("")) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create UYVY encoder");
	}
}

UVUyvy::~UVUyvy() noexcept
{
}


bool UVUyvy::set_param(const core::Parameter& param)
{
	return ultragrid::UVVideoCompress::set_param(param);
}

} /* namespace uv_jpeg_compress */
} /* namespace yuri */
