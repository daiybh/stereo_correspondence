/*!
 * @file 		UnixStreamSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UNIXSTREAMSOCKET_H_
#define UNIXSTREAMSOCKET_H_

#include "yuri/core/socket/StreamSocket.h"
#include "YuriStreamSocket.h"
namespace yuri {
namespace network {

class UnixStreamSocket: public YuriStreamSocket
{
public:
	UnixStreamSocket(const log::Log &log_);
	UnixStreamSocket(const log::Log &log_, int);
	virtual ~UnixStreamSocket() noexcept;
private:

	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;
	virtual core::socket::pStreamSocket prepare_new(int sock_raw) override;
};

} /* namespace yuri_tcp */
} /* namespace yuri */



#endif /* UNIXSTREAMSOCKET_H_ */
