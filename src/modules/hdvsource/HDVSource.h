/*
 * HDVSource.h
 *
 *  Created on: May 29, 2009
 *      Author: neneko
 */

#ifndef HDVSOURCE_H_
#define HDVSOURCE_H_

#include "yuri/ieee1394/IEEE1394SourceBase.h"
#include "yuri/config/RegisteredClass.h"

namespace yuri {

namespace io {

class HDVSource:
	public IEEE1394SourceBase
{
public:
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters);
	static shared_ptr<Parameters> configure();

	HDVSource(Log &log_,pThreadBase parent, nodeid_t node=0, int port = 0, int64_t guid=-1);
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
