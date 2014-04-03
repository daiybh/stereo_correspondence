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

	virtual void 		run();


	AVFormatContext*	fmtctx;
	std::vector<AVCodec*>
						video_codecs_;
	std::vector<AVCodec*>
						audio_codecs_;
	yuri::size_t 		block;
	std::string 		filename;
	std::vector<AVStream*>
						video_streams_;
	std::vector<AVStream*>
						audio_streams_;
	std::vector<format_t>
						video_formats_;
	std::vector<format_t>
						video_formats_out_;
	std::vector<format_t>
						audio_formats_;
	std::vector<format_t>
						audio_formats_out_;
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

};

} /* namespace video */
} /* namespace yuri */
#endif /* AVDEMUXER_H_ */
