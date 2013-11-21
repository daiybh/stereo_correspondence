/*!
 * @file 		DVSource.h
 * @author 		Zdenek Travnicek
 * @date 		16.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DVSOURCE_H_
#define DVSOURCE_H_

#include "yuri/ieee1394/IEEE1394SourceBase.h"
#include "yuri/core/frame/CompressedVideoFrame.h"

namespace yuri {

namespace ieee1394 {

class DVSource: public IEEE1394SourceBase {
public:
	DVSource(const log::Log &log_,core::pwThreadBase parent, nodeid_t node=0, int port = 0, int64_t guid=-1);
	virtual ~DVSource() noexcept;

//	static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters);
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
protected:
	virtual bool start_receiving();
	virtual bool stop_receiving();
	static int receive_frame (unsigned char *data, int len, int complete, void *callback_data);
	int process_frame(unsigned char *data, int length, int complete);
	bool analyze_frame(core::pCompressedVideoFrame& frame);

protected:
	iec61883_dv_fb_t frame;
};

}

}
#endif /* DVSOURCE_H_ */
