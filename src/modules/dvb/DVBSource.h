/*
 * DVBSource.h
 *
 *  Created on: 20.2.2011
 *      Author: neneko
 */

#ifndef DVBSOURCE_H_
#define DVBSOURCE_H_

#include "yuri/core/thread/IOThread.h"
namespace yuri {

namespace dvb {

class DVBSource: public core::IOThread {

public:
	DVBSource(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters);
	virtual ~DVBSource() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();

private:
	virtual bool set_param(const core::Parameter& param) override;
	virtual void run() override;
	virtual bool step() override;
	bool init();
	std::string adapter_path_;
	std::string frontend_;
	int frequency_;
	int pid_;
	int apid_;
	size_t chunk_size_;


	int fe_fd;
	int demux_fd, ademux_fd;
	int dvr_fd;
	bool output_ts_;
};

}

}
#endif /* DVBSOURCE_H_ */
