/*!
 * @file 		HDVSource.h
 * @author 		Zdenek Travnicek
 * @date 		29.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef HDVSOURCE_H_
#define HDVSOURCE_H_

#include "yuri/ieee1394/IEEE1394SourceBase.h"

namespace yuri {

namespace ieee1394 {

class HDVSource:
	public IEEE1394SourceBase
{
public:
	static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	static core::pParameters configure();

	HDVSource(log::Log &log_,core::pwThreadBase parent, nodeid_t node=0, int port = 0, int64_t guid=-1);
	virtual ~HDVSource();
	//virtual void run();
	void setOutputBufferSize(long size);
protected:
	virtual bool start_receiving();
	virtual bool stop_receiving();
	static int receive_frame (unsigned char *data, int len, unsigned int dropped, void *callback_data);
	int process_frame(unsigned char *data, int len, unsigned int dropped);

	void do_sendOutputBuffer();
	void do_send_data(yuri::ubyte_t*, yuri::size_t size);
	iec61883_mpeg2_t mpeg_frame;
	std::map<int,int> counters;
	long total_packets, total_missing;
	long buffer_size, buffer_position;
	shared_array<yuri::ubyte_t> output_buffer;
	bool enable_checks;
	boost::mutex buffer_lock;
};

}

}

#endif /* HDVSOURCE_H_ */
