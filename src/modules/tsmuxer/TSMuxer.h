/*
 * TSMuxer.h
 *
 *  Created on: Aug 10, 2010
 *      Author: worker
 */

#ifndef TSMUXER_H_
#define TSMUXER_H_

#include "yuri/libav/AVCodecBase.h"
#include "yuri/config/Config.h"
#include "yuri/config/RegisteredClass.h"
#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/smart_ptr.hpp>
extern "C" {
	#include <libavformat/avformat.h>
}

namespace yuri {

namespace video {

using boost::shared_ptr;
using namespace boost::posix_time;

class TSMuxer:public AVCodecBase {
public:
	TSMuxer(Log &_log, pThreadBase parent);
	virtual ~TSMuxer();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();

	bool step();
protected:
	AVOutputFormat *output_format;
	shared_ptr<AVFormatContext> format_context;
	vector<shared_ptr<AVStream> > streams;
	shared_array<yuri::ubyte_t> buffer;
	shared_ptr<AVIOContext> byte_context;
	yuri::size_t buffer_size, buffer_position;
	yuri::size_t pts, dts, duration;

protected:
	bool reconfigure();
	bool reconfigure(shared_ptr<BasicFrame> frame);
	bool put_frame(shared_ptr<BasicFrame> frame);
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

