/*
 * TSRtpReceiver.h
 *
 *  Created on: Feb 22, 2011
 *      Author: worker
 */

#ifndef TSRTPRECEIVER_H_
#define TSRTPRECEIVER_H_
#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include "yuri/asio/ASIOUDPSocket.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace io {


struct RTPPacket {
	yuri::ubyte_t bytes[4];

	//unsigned int sequence:16;
	//yuri::size_t timestamp:32;
	//yuri::size_t SSRC:32;
	yuri::uint_t timestamp:32;
	yuri::uint_t SSRC:32;
	//unsigned int CSRC:32;
} __attribute__ ((packed));
using namespace boost::posix_time;

class TSRtpReceiver: public BasicIOThread {
public:
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();

	TSRtpReceiver(Log &log_, pThreadBase parent, Parameters& parameters);
	virtual ~TSRtpReceiver();

	bool set_endpoint(std::string address, yuri::size_t port);
	yuri::size_t seq, pseq;
	void run();
protected:
	shared_ptr<ASIOUDPSocket> socket;
	ptime first_packet;
	yuri::size_t packets_received;
	shared_array<yuri::ubyte_t> buffer, in_buffer;
	yuri::size_t buffer_size, buffer_position;
	bool pass_thru;
};

}

}

#endif /* TSRTPRECEIVER_H_ */
