/*!
 * @file 		YuriTcp.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		27.10.2013
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriTcp.h"
#include "yuri/core/utils.h"
#include <sys/socket.h>
#include "inet_utils.h"

namespace yuri {
namespace network {


YuriTcp::YuriTcp(const log::Log &log_):
YuriStreamSocket(log_, AF_INET)
{
}
YuriTcp::YuriTcp(const log::Log &log_, int sock):
YuriStreamSocket(log_, AF_INET, sock)
{

}
YuriTcp::~YuriTcp() noexcept
{
}

bool YuriTcp::do_bind(const std::string& url, uint16_t port)
{
	return inet::bind(socket_, url, port);
}
bool YuriTcp::do_connect(const std::string& address, uint16_t port)
{
	return inet::connect(socket_, address, port);
}

core::socket::pStreamSocket YuriTcp::prepare_new(int sock_raw)
{
	if (sock_raw > 0) return std::make_shared<YuriTcp>(log, sock_raw);
	return {};
}



} /* namespace network */
} /* namespace yuri */
