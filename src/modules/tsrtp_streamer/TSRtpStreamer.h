/*!
 * @file 		TSRtpStreamer.h
 * @author 		Zdenek Travnicek
 * @date 		15.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef TSRTPSTREAMER_H_
#define TSRTPSTREAMER_H_

#include "yuri/core/IOThread.h"
#include "yuri/asio/ASIOUDPSocket.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace rtp {
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
	uint8_t bytes[4];

	//unsigned int sequence:16;
	uint32_t timestamp;
	uint32_t SSRC;
	//unsigned int CSRC:32;
};
using namespace boost::posix_time;
class TSRtpStreamer: public yuri::core::IOThread {
public:
	TSRtpStreamer(log::Log &log_, core::pwThreadBase parent);
	virtual ~TSRtpStreamer();
	static core::pIOThread generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters);
	static core::pParameters configure();
	bool set_endpoint(std::string address, yuri::size_t port);

	bool step();
	yuri::size_t seq, pseq;
protected:
	shared_ptr<asio::ASIOUDPSocket> socket;
	ptime first_packet;
	yuri::size_t packets_sent;

};

}

}

#endif /* TSRTPSTREAMER_H_ */
