/*
 * UVLibavDecompress.cpp
 *
 *  Created on: 10.10.2014
 *      Author: neneko
 */

#include "UVLibavDecompress.h"
#include "yuri/core/Module.h"
#include "YuriUltragrid.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"

extern "C" {
#include "video_decompress.h"
#include "video_decompress/libavcodec.h"
#include "uv_video.h"
}

namespace yuri {
namespace uv_libav {

IOTHREAD_GENERATOR(UVLibavDecompress)

core::Parameters UVLibavDecompress::configure()
{
	core::Parameters p = ultragrid::UVVideoDecompress::configure();
	p.set_description("UVLibav");
	return p;
}


UVLibavDecompress::UVLibavDecompress(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoDecompress(log_,parent, "uv_libav", UV_DECOMPRESS_DETAIL(libavcodec))
{
	IOTHREAD_INIT(parameters)

	if(!init_decompressor({})) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create libavcodec encoder");
	}
}

UVLibavDecompress::~UVLibavDecompress() noexcept
{
}


}
}



