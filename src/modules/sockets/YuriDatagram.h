/*!
 * @file 		YuriDatagram.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURIDATAGRAM_H_
#define YURIDATAGRAM_H_

#include "yuri/core/socket/DatagramSocket.h"
#include "YuriNetSocket.h"

namespace yuri {
namespace network {

class YuriDatagram: public core::socket::DatagramSocket
{
public:
	YuriDatagram(const log::Log &log_, const std::string&, int domain);
	virtual ~YuriDatagram() noexcept;
protected:
	int get_socket() { return socket_.get_socket(); }
private:

	virtual size_t do_send_datagram(const uint8_t* data, size_t size) override;
	virtual size_t do_receive_datagram(uint8_t* data, size_t size) override;
	virtual bool do_ready_to_send() override;

	virtual bool do_data_available() override;
	virtual bool do_wait_for_data(duration_t duration) override;
protected:
	YuriNetSocket socket_;
};

}
}




#endif /* YUVIDATAGRAM_H_ */
