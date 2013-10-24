/*
 * UVVideoCompress.h
 *
 *  Created on: 24.10.2013
 *      Author: neneko
 */

#ifndef UVVIDEOCOMPRESS_H_
#define UVVIDEOCOMPRESS_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
extern "C" {
#include "video_compress.h"
#include "video_frame.h"
}
struct module;
namespace yuri {
namespace ultragrid {

namespace detail {
typedef function<module* (module *, const video_compress_params *)>	compress_init_t;
typedef function<video_frame *(module *, video_frame *, int)> 	compress_t;

struct uv_video_compress_params {
	std::string 			name;
	compress_init_t			init_func;
	compress_t				compress_func;
};

#define UV_COMPRESS_DETAIL(name) \
yuri::ultragrid::detail::uv_video_compress_params { \
	#name, \
	name ## _compress_init, \
	name ## _compress, \
}

}


class UVVideoCompress: public core::SpecializedIOFilter<core::RawVideoFrame>
{
public:
	UVVideoCompress(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_video_compress_params uv_compress_params);
	virtual ~UVVideoCompress() noexcept;
protected:
	bool init_compressor(const std::string& params);
private:

	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	module* encoder_;
	detail::uv_video_compress_params uv_compress_params_;

};
}

}


#endif /* UVVIDEOCOMPRESS_H_ */
