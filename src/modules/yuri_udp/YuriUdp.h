/*!
 * @file 		YuriUdp.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.11.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURIUDP_H_
#define YURIUDP_H_

#include "yuri/core/socket/DatagramSocket.h"

namespace yuri {
namespace yuri_udp {

class YuriUdp: public core::socket::DatagramSocket
{
public:
	YuriUdp(const log::Log &log_, const std::string&);
	virtual ~YuriUdp() noexcept;
private:
	
	virtual size_t do_send_datagram(const uint8_t* data, size_t size);
	virtual size_t do_receive_datagram(uint8_t* data, size_t size);
	virtual bool do_ready_to_send();

	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;

	virtual bool do_data_available() override;
	virtual bool do_wait_for_data(duration_t duration) override;
	int socket_;
};

} /* namespace yuri_tcp */
} /* namespace yuri */
#endif /* YURIUDP_H_ */
