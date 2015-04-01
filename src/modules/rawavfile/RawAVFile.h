/*
 * RawAVFile.h
 *
 *  Created on: Feb 9, 2012
 *      Author: neneko
 */

#ifndef AVDEMUXER_H_
#define AVDEMUXER_H_
#include "yuri/libav/libav.h"
#include "yuri/core/thread/IOFilter.h"
#include "yuri/core/utils/managed_resource.h"
extern "C" {
	#include <libavformat/avformat.h>
}

#include <vector>

namespace yuri {
namespace rawavfile {


class RawAVFile: public core::IOThread {
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	RawAVFile(const log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~RawAVFile() noexcept;
	virtual bool 		set_param(const core::Parameter &param);
private:

	virtual void 		run() override;

	bool				open_file(const std::string& filename);
	bool				push_ready_frames();
	bool				process_file_end();

	bool				process_undecoded_frame(index_t idx, const AVPacket& packet);
	bool				decode_video_frame(index_t idx, const AVPacket& packet, AVFrame* av_frame, bool& keep_packet);
	core::utils::managed_resource<AVFormatContext>
						fmtctx_;


	struct stream_detail_t {
		stream_detail_t(AVStream* stream=nullptr, AVCodec* codec = nullptr, format_t fmt = 0, format_t fmt_out = 0)
		:stream(stream),ctx(stream?stream->codec:nullptr),codec(codec),format(fmt),format_out(fmt_out) {}
		AVStream* stream;
		AVCodecContext *ctx;
		AVCodec* codec;
		format_t format;
		format_t format_out;
		resolution_t resolution;
		duration_t delta;
	};
	std::string 		filename_;

	std::vector<stream_detail_t>
						video_streams_;
	std::vector<stream_detail_t>
						audio_streams_;

	format_t 			format_out_;
	format_t 			video_format_out_;
	bool				decode_;
	double 				fps_;
	std::vector<timestamp_t>
						next_times_;
	std::vector<core::pFrame>
						frames_;
	size_t 				max_video_streams_;
	size_t 				max_audio_streams_;

	bool 				loop_;
};

} /* namespace video */
} /* namespace yuri */
#endif /* AVDEMUXER_H_ */
