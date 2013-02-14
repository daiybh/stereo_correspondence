/*
 * TSRtpStreamer.h
 *
 *  Created on: Aug 15, 2010
 *      Author: neneko
 */

#ifndef TSRTPSTREAMER_H_
#define TSRTPSTREAMER_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include "yuri/asio/ASIOUDPSocket.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace io {
/*
struct RTPPacket {
	unsigned int version:2;
	unsigned int padding:1;
	unsigned int extension:1;
	unsigned int CC:4;
	unsigned int marker:1;
	unsigned int payload_type:7;
	unsigned int sequence:16;
	unsigned int timestamp:32;
	unsigned int SSRC:32;
	unsigned int CSRC:32;
};
*/

struct RTPPacket {
	char bytes[4];

	//unsigned int sequence:16;
	unsigned timestamp:32;
	unsigned int SSRC:32;
	//unsigned int CSRC:32;
};
using namespace boost::posix_time;
class TSRtpStreamer: public yuri::io::BasicIOThread {
public:
	TSRtpStreamer(Log &log_, pThreadBase parent);
	virtual ~TSRtpStreamer();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
	bool set_endpoint(std::string address, yuri::size_t port);

	bool step();
	yuri::size_t seq, pseq;
protected:
	shared_ptr<ASIOUDPSocket> socket;
	ptime first_packet;
	yuri::size_t packets_sent;

};

}

}

#endif /* TSRTPSTREAMER_H_ */
