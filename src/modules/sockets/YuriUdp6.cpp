/*!
 * @file 		YuriUdp6.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		20.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriUdp6.h"
#include "inet_utils.h"
#include <sys/socket.h>

namespace yuri {
namespace network {


YuriUdp6::YuriUdp6(const log::Log &log_, const std::string& s):
YuriDatagram(log_, s, AF_INET6)
{
}

YuriUdp6::~YuriUdp6() noexcept
{
}

bool YuriUdp6::do_bind(const std::string& url, uint16_t port)
{
	return inet::bind(socket_, url, port);
}
bool YuriUdp6::do_connect(const std::string& address, uint16_t port)
{
	return inet::connect(socket_, address, port);
}

} /* namespace network */
} /* namespace yuri */
