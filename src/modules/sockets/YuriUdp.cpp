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
#include "yuri/core/utils.h"
#include <sys/types.h>
#include <sys/socket.h>
//#include <linux/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <netdb.h>

namespace yuri {
namespace network {


YuriUdp::YuriUdp(const log::Log &log_, const std::string& s):
YuriDatagram(log_, s, AF_INET)
{
}

YuriUdp::~YuriUdp() noexcept
{
}

namespace {
unique_ptr<addrinfo, function<void(addrinfo*)>>
get_addr_info(const std::string& server, uint16_t port)
{
	static const addrinfo hints = {AI_PASSIVE, AF_UNSPEC, SOCK_STREAM, 0, 0, nullptr, nullptr, nullptr};
	addrinfo *info = nullptr;
	const char* addr = nullptr;
	if (!server.empty()) addr = server.c_str();
	/*int ret = */::getaddrinfo(addr,
					lexical_cast<std::string>(port).c_str(),
	                       &hints,
	                       &info);
	return {info, [](addrinfo* p){freeaddrinfo(p);}};
}

}

bool YuriUdp::do_bind(const std::string& address, uint16_t port)
{
	auto info = get_addr_info(address, port);
	if (!info) return false;
	return ::bind(get_socket(), info->ai_addr, info->ai_addrlen) == 0;
}

bool YuriUdp::do_connect(const std::string& address, uint16_t port)
{
	auto info = get_addr_info(address, port);
	if (!info) return false;
	return ::connect(get_socket(), info->ai_addr, info->ai_addrlen) == 0;
}
} /* namespace network */
} /* namespace yuri */
