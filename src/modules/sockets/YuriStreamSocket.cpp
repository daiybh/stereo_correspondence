/*!
 * @file 		YuriStreamSocket.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriStreamSocket.h"
#include <sys/socket.h>
namespace yuri {
namespace network {


YuriStreamSocket::YuriStreamSocket(const log::Log &log_, int domain):
core::socket::StreamSocket(log_),
socket_(domain, SOCK_STREAM, 0)
{

}
YuriStreamSocket::YuriStreamSocket(const log::Log &log_, int /* domain */, int sock_raw):
core::socket::StreamSocket(log_),
socket_(sock_raw)
{

}

YuriStreamSocket::~YuriStreamSocket() noexcept
{
	::shutdown(get_socket(),2);
}


size_t YuriStreamSocket::do_send_data(const uint8_t* data, size_t data_size)
{
	ssize_t wrote = ::send(get_socket(), data, data_size, 0);
	return (wrote>0)?wrote:0;
}
size_t YuriStreamSocket::do_receive_data(uint8_t* data, size_t size)
{
	ssize_t read = ::recv(get_socket(), data, size, MSG_DONTWAIT);
	return (read>0)?read:0;
}

bool YuriStreamSocket::do_listen()
{
	return ::listen(get_socket(), 10) == 0;
}
core::socket::pStreamSocket YuriStreamSocket::do_accept()
{
	auto sock = ::accept(get_socket(), nullptr, 0);
	return prepare_new(sock);
}

bool YuriStreamSocket::do_data_available()
{
	return socket_.data_available();
}
bool YuriStreamSocket::do_wait_for_data(duration_t duration)
{
	return socket_.wait_for_data(duration);
}

}
}
