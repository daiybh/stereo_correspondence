/*!
 * @file 		NullSockets.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef NULL_SOCKETS_H_
#define NULL_SOCKETS_H_

#include "DatagramSocket.h"
#include "StreamSocket.h"
#include "yuri/core/thread/ThreadBase.h"

namespace yuri {
namespace core {
namespace socket {

class NullDatagramSocket: public DatagramSocket
{
public:
	EXPORT NullDatagramSocket(const log::Log& log_, const std::string& url);
	virtual ~NullDatagramSocket() noexcept {}
private:
	EXPORT virtual size_t do_send_datagram(const uint8_t*, size_t size) override {
		return size;
	}
	EXPORT virtual size_t do_receive_datagram(uint8_t*, size_t) override {
		return 0;
	}
	EXPORT virtual bool do_bind(const std::string&, port_t) override
	{
		return true;
	}
	EXPORT virtual bool do_connect(const std::string&, port_t) override {
		return true;
	}
	EXPORT virtual bool do_data_available() override {
		return false;
	}
	EXPORT virtual bool do_ready_to_send() override {
		return false;
	}
	EXPORT virtual bool do_wait_for_data(duration_t duration) override {
		core::ThreadBase::sleep(duration);
		return false;
	}

};


class NullStreamSocket : public StreamSocket
{
public:
	EXPORT NullStreamSocket(const log::Log& log_);
	EXPORT virtual ~NullStreamSocket() noexcept {}
private:
	EXPORT virtual size_t do_send_data(const uint8_t*, size_t size) override {
		return size;
	}
	EXPORT virtual size_t do_receive_data(uint8_t* , size_t) override {
		return 0;
	}
	EXPORT virtual bool do_bind(const std::string& , uint16_t) override  {
		return true;
	}
	EXPORT virtual bool do_connect(const std::string&, uint16_t) override {
		return true;
	}
	EXPORT virtual bool do_listen() override {
		return true;
	}
	EXPORT virtual pStreamSocket do_accept() override {
		return{};
	}

	EXPORT virtual bool do_data_available() override {
		return false;
	}
	EXPORT virtual bool do_wait_for_data(duration_t duration) override {
		core::ThreadBase::sleep(duration);
		return false;
	}


};

}
}
}

#endif 
