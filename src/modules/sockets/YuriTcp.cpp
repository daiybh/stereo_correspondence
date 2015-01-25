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
#include <sys/types.h>
#include <sys/socket.h>
//#include <linux/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <netdb.h>

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

bool YuriTcp::do_bind(const std::string& address, uint16_t port)
{
	auto info = get_addr_info(address, port);
	if (!info) return false;
	return ::bind(get_socket(), info->ai_addr, info->ai_addrlen) == 0;
}

bool YuriTcp::do_connect(const std::string& address, uint16_t port)
{
	auto info = get_addr_info(address, port);
	if (!info) return false;
	return ::connect(get_socket(), info->ai_addr, info->ai_addrlen) == 0;
}
core::socket::pStreamSocket YuriTcp::prepare_new(int sock_raw)
{
	if (sock_raw > 0) return std::make_shared<YuriTcp>(log, sock_raw);
	return {};
}

} /* namespace network */
} /* namespace yuri */
