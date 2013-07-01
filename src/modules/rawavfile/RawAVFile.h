/*
 * RawAVFile.h
 *
 *  Created on: Feb 9, 2012
 *      Author: neneko
 */

#ifndef AVDEMUXER_H_
#define AVDEMUXER_H_
#include "yuri/libav/AVCodecBase.h"
extern "C" {
	#include <libavformat/avformat.h>
}
#include <vector>
//#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {
namespace video {

class RawAVFile: public core::BasicIOThread, public AVCodecBase {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~RawAVFile();
	virtual bool 		set_param(const core::Parameter &param);
private:
	RawAVFile(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
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
	format_t 			format_out_;
	format_t 			video_format_out_;
	bool				decode_;
	double 				fps_;
	std::vector<time_value>
						next_times_;
	std::vector<core::pBasicFrame>
						frames_;
	size_t 				max_video_streams_;
	size_t 				max_audio_streams_;

};

} /* namespace video */
} /* namespace yuri */
#endif /* AVDEMUXER_H_ */
