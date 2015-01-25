/*!
 * @file 		YuriTcp.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.10.2013
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURITCP_H_
#define YURITCP_H_

#include "yuri/core/socket/StreamSocket.h"
#include "YuriStreamSocket.h"
namespace yuri {
namespace network {

class YuriTcp: public YuriStreamSocket
{
public:
	YuriTcp(const log::Log &log_);
	YuriTcp(const log::Log &log_, int);
	virtual ~YuriTcp() noexcept;
private:

	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;
	virtual core::socket::pStreamSocket prepare_new(int sock_raw) override;
};

} /* namespace network */
} /* namespace yuri */
#endif /* YURITCP_H_ */
