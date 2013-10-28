/*
 * ReadPcap.h
 *
 *  Created on: 26.2.2013
 *      Author: neneko
 */

#ifndef READPCAP_H_
#define READPCAP_H_

#include "yuri/core/IOThread.h"
#include <fstream>

namespace yuri {
namespace pcap {

class ReadPcap: public core::IOThread
{
public:
	virtual ~ReadPcap();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
private:
	ReadPcap(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual void run();
	virtual bool set_param(const core::Parameter &param);
	std::string filename_;
	std::ifstream file_;
	size_t packet_count;
	size_t last_valid_size;
};

} /* namespace pcap */
} /* namespace yuri */


#endif /* READPCAP_H_ */
