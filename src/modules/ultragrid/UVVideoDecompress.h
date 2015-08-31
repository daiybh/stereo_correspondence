/*
 * UVVideoDecompress.h
 *
 *  Created on: 10.10.2014
 *      Author: neneko
 */

#ifndef UVVIDEODECOMPRESS_H_
#define UVVIDEODECOMPRESS_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"

#include "yuri/core/thread/ConverterThread.h"
#include "video_decompress.h"
#include "uv_video.h"

namespace yuri {
namespace ultragrid {


namespace detail {
typedef std::function<void* ()>	uv_decompress_init_t;

typedef std::function<int (void *state,
					unsigned char *dst,
					unsigned char *buffer,
					unsigned int src_len,
					int frame_seq)> 	uv_decompress_t;

typedef std::function<void (void*)>	uv_decompress_done_t;

typedef std::function<int (void *, struct video_desc desc, int, int, int, int, codec_t)> uv_decompress_reconfigure_t;

struct uv_video_decompress_params {
	std::string 			name;
	uv_decompress_init_t		init_func;
	uv_decompress_t			decompress_func;
	uv_decompress_done_t		decompress_done;
	uv_decompress_reconfigure_t decompress_reconfigure;

};

}

#define UV_DECOMPRESS_DETAIL(name) \
yuri::ultragrid::detail::uv_video_decompress_params { \
	#name, \
	name ## _decompress_init, \
	name ## _decompress, \
	name ## _decompress_done, \
	name ## _decompress_reconfigure, \
}


class UVVideoDecompress: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public core::ConverterThread
{
	using base_type = core::SpecializedIOFilter<core::CompressedVideoFrame>;
public:
	UVVideoDecompress(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_video_decompress_params uv_decompress_params);
	virtual ~UVVideoDecompress() noexcept;
	static core::Parameters configure();

protected:
	bool init_decompressor(const std::string& params);
private:

	virtual core::pFrame do_special_single_step(core::pCompressedVideoFrame frame) override;
	virtual core::pFrame 		do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool do_converter_is_stateless() const override { return false; }
	virtual bool 				set_param(const core::Parameter& param) override;

	void* decoder_;
	detail::uv_video_decompress_params uv_decompress_params_;

	resolution_t last_resolution_;
	format_t last_format_;
	format_t output_format_;
	size_t sequence_;


};




}
}


#endif /* UVVIDEODECOMPRESS_H_ */
