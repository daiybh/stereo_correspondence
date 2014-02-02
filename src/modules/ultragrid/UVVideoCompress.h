/*!
 * @file 		UVVideoCompress.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		24.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVVIDEOCOMPRESS_H_
#define UVVIDEOCOMPRESS_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
extern "C" {
#include "video_compress.h"
#include "video_frame.h"
}
struct module;
namespace yuri {
namespace ultragrid {

namespace detail {
typedef function<module* (module *, const video_compress_params *)>	compress_init_t;
typedef function<video_frame* (module *, video_frame *, int)> 	compress_t;
typedef function<video_frame* (module *, video_frame *, int, int)> compress_tile_t;

struct uv_video_compress_params {
	std::string 			name;
	compress_init_t			init_func;
	compress_t				compress_func;
	compress_tile_t			compress_tile_func;
};

#define UV_COMPRESS_DETAIL(name) \
yuri::ultragrid::detail::uv_video_compress_params { \
	#name, \
	name ## _compress_init, \
	name ## _compress, \
	{} \
}

#define UV_COMPRESS_DETAIL_TILE(name) \
yuri::ultragrid::detail::uv_video_compress_params { \
	#name, \
	name ## _compress_init, \
	{}, \
	name ## _compress_tile, \
}

}


class UVVideoCompress: public core::SpecializedIOFilter<core::RawVideoFrame>, core::ConverterThread
{
public:
	UVVideoCompress(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_video_compress_params uv_compress_params);
	virtual ~UVVideoCompress() noexcept;
protected:
	bool init_compressor(const std::string& params);
private:

	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual core::pFrame 		do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	module* encoder_;
	detail::uv_video_compress_params uv_compress_params_;
	unique_ptr<video_frame, void(*)(video_frame*)> uv_frame;
};
}

}


#endif /* UVVIDEOCOMPRESS_H_ */
