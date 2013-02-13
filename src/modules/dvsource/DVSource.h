/*
 * DVSource.h
 *
 *  Created on: Jul 16, 2009
 *      Author: neneko
 */

#ifndef DVSOURCE_H_
#define DVSOURCE_H_

#include "yuri/ieee1394/IEEE1394SourceBase.h"
#include "yuri/config/RegisteredClass.h"

namespace yuri {

namespace io {
using namespace yuri::config;
class DVSource: public yuri::io::IEEE1394SourceBase {
public:
	DVSource(Log &log_,pThreadBase parent, nodeid_t node=0, int port = 0, int64_t guid=-1);
	virtual ~DVSource();

	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters);
	static shared_ptr<Parameters> configure();
protected:
	virtual bool start_receiving();
	virtual bool stop_receiving();
	static int receive_frame (unsigned char *data, int len, int complete, void *callback_data);
	int process_frame(unsigned char *data, int length, int complete);
	bool analyze_frame(shared_ptr<BasicFrame> frame);

protected:
	iec61883_dv_fb_t frame;
};

}

}
#endif /* DVSOURCE_H_ */
