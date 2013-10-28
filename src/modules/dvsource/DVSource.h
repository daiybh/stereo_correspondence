/*!
 * @file 		DVSource.h
 * @author 		Zdenek Travnicek
 * @date 		16.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DVSOURCE_H_
#define DVSOURCE_H_

#include "yuri/ieee1394/IEEE1394SourceBase.h"


namespace yuri {

namespace ieee1394 {

class DVSource: public IEEE1394SourceBase {
public:
	DVSource(log::Log &log_,core::pwThreadBase parent, nodeid_t node=0, int port = 0, int64_t guid=-1);
	virtual ~DVSource();

	static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters);
	static core::pParameters configure();
protected:
	virtual bool start_receiving();
	virtual bool stop_receiving();
	static int receive_frame (unsigned char *data, int len, int complete, void *callback_data);
	int process_frame(unsigned char *data, int length, int complete);
	bool analyze_frame(core::pBasicFrame frame);

protected:
	iec61883_dv_fb_t frame;
};

}

}
#endif /* DVSOURCE_H_ */
