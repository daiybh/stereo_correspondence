/*!
 * @file 		YuriUdp.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		27.11.2014
 * @date		25.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2014 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriUdp.h"
#include "inet_utils.h"
#include <sys/socket.h>

namespace yuri {
namespace network {


YuriUdp::YuriUdp(const log::Log &log_, const std::string& s):
YuriDatagram(log_, s, AF_INET)
{
}

YuriUdp::~YuriUdp() noexcept
{
}

bool YuriUdp::do_bind(const std::string& url, uint16_t port)
{
	return inet::bind(socket_, url, port);
}
bool YuriUdp::do_connect(const std::string& address, uint16_t port)
{
	return inet::connect(socket_, address, port);
}

} /* namespace network */
} /* namespace yuri */
