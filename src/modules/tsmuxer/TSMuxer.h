/*!
 * @file 		TSMuxer.h
 * @author 		Zdenek Travnicek
 * @date 		10.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef TSMUXER_H_
#define TSMUXER_H_

#include "yuri/libav/AVCodecBase.h"
#include <boost/date_time/posix_time/posix_time.hpp>
extern "C" {
	#include <libavformat/avformat.h>
}

namespace yuri {

namespace video {

class TSMuxer:public AVCodecBase {
public:
	TSMuxer(log::Log &_log, core::pwThreadBase parent);
	virtual ~TSMuxer();
	static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	static core::pParameters configure();

	bool step();
protected:
	AVOutputFormat *output_format;
	shared_ptr<AVFormatContext> format_context;
	std::vector<shared_ptr<AVStream> > streams;
	//shared_array<yuri::ubyte_t> buffer;
	std::vector<yuri::ubyte_t> buffer;
	shared_ptr<AVIOContext> byte_context;
	yuri::size_t buffer_size, buffer_position;
	yuri::size_t pts, dts, duration;

protected:
	bool reconfigure();
	bool reconfigure(core::pBasicFrame frame);
	bool put_frame(core::pBasicFrame frame);
	static int read_buffer(void *opaque, yuri::ubyte_t *buf, int size);
	static int write_buffer(void *opaque, yuri::ubyte_t *buf, int size);
	static int64_t seek_buffer(void *opaque, int64_t offset, int whence);
	int _read_buffer(yuri::ubyte_t *buf, int size);
	int _write_buffer(yuri::ubyte_t *buf, int size);
	int64_t _seek_buffer(int64_t offset, int whence);
};

}

}
#endif /* TSMUXER_H_ */

