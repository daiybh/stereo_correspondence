/*!
 * @file 		YuriTcp.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		27.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriTcp.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/core/utils.h"
#include <sys/types.h>
#include <sys/socket.h>
//#include <linux/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <netdb.h>

namespace yuri {
namespace yuri_tcp {


MODULE_REGISTRATION_BEGIN("yuri_tcp")
	REGISTER_STREAM_SOCKET("yuri_tcp",YuriTcp)
MODULE_REGISTRATION_END()


YuriTcp::YuriTcp(const log::Log &log_):
core::socket::StreamSocket(log_)
{
	socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
}

YuriTcp::~YuriTcp() noexcept
{
	::close(socket_);
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
	return ::bind(socket_, info->ai_addr, info->ai_addrlen) == 0;
}

size_t YuriTcp::do_send_data(const uint8_t* data, size_t data_size)
{
	ssize_t wrote = ::send(socket_, data, data_size, 0);
	return (wrote>0)?wrote:0;
}
size_t YuriTcp::do_receive_data(uint8_t* data, size_t size)
{
	ssize_t read = ::recv(socket_, data, size, MSG_DONTWAIT);
	return (read>0)?read:0;
}

bool YuriTcp::do_connect(const std::string& address, uint16_t port)
{
	auto info = get_addr_info(address, port);
	if (!info) return false;
	return ::connect(socket_, info->ai_addr, info->ai_addrlen) == 0;
}

bool YuriTcp::do_data_available()
{
	pollfd fds = {socket_, POLLIN, 0};
	::poll(&fds, 1, 0);
	return (fds.revents & POLLIN);
}
bool YuriTcp::do_wait_for_data(duration_t duration)
{
	pollfd fds = {socket_, POLLIN, 0};
	::poll(&fds, 1, static_cast<int>(duration.value/1000));
	return (fds.revents & POLLIN);
}

} /* namespace yuri_tcp */
} /* namespace yuri */
