/*!
 * @file 		YuriStreamSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURISTREAMSOCKET_H_
#define YURISTREAMSOCKET_H_
#include "yuri/core/socket/StreamSocket.h"
#include "YuriNetSocket.h"

namespace yuri {
namespace network {

class YuriStreamSocket: public core::socket::StreamSocket
{
public:
	YuriStreamSocket(const log::Log &log_, int domain);
	YuriStreamSocket(const log::Log &log_, int domain, int sock_raw);
	virtual ~YuriStreamSocket() noexcept;
protected:
	int get_socket() { return socket_.get_socket(); }

private:

	virtual size_t do_send_data(const uint8_t* data, size_t data_size) override;
	virtual size_t do_receive_data(uint8_t* data, size_t size) override;
	virtual bool do_listen() override;
	virtual core::socket::pStreamSocket do_accept() override;

	virtual core::socket::pStreamSocket prepare_new(int sock_raw) = 0;

	virtual bool do_data_available() override;
	virtual bool do_wait_for_data(duration_t duration) override;
protected:
	YuriNetSocket socket_;
};

}
}

#endif /* YURISTREAMSOCKET_H_ */
