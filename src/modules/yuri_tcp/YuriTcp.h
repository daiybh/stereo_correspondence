/*!
 * @file 		YuriTcp.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURITCP_H_
#define YURITCP_H_

#include "yuri/core/socket/StreamSocket.h"

namespace yuri {
namespace yuri_tcp {

class YuriTcp: public core::socket::StreamSocket
{
public:
	YuriTcp(const log::Log &log_);
	YuriTcp(const log::Log &log_, int);
	virtual ~YuriTcp() noexcept;
private:

	virtual size_t do_send_data(const uint8_t* data, size_t data_size) override;
	virtual size_t do_receive_data(uint8_t* data, size_t size) override;
	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;
	virtual bool do_listen() override;
	virtual core::socket::pStreamSocket do_accept() override;

	virtual bool do_data_available() override;
	virtual bool do_wait_for_data(duration_t duration) override;
	int socket_;
};

} /* namespace yuri_tcp */
} /* namespace yuri */
#endif /* YURITCP_H_ */
