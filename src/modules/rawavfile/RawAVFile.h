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
namespace yuri {
namespace video {

class RawAVFile: public AVCodecBase {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~RawAVFile();
	virtual bool 		set_param(const core::Parameter &param);
private:
	RawAVFile(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual void 		run();
	AVFormatContext* 	fmtctx;
	AVCodec*			video_codec;
	yuri::size_t 		block;
	std::string 		filename;
	AVStream* 			video_stream;
	format_t 			format_;
	format_t 			format_out_;
	bool				decode_;
	double 				fps_;

};

} /* namespace video */
} /* namespace yuri */
#endif /* AVDEMUXER_H_ */
