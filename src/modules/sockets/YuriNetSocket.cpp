/*!
 * @file 		YuriNetSocket.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriNetSocket.h"
#include "poll.h"
#include <sys/socket.h>
#include <unistd.h>
namespace yuri {
namespace network {

YuriNetSocket::YuriNetSocket(int domain, int type, int proto)
{
	socket_ = ::socket(domain, type, proto);
	int i = 1;
	setsockopt(socket_,SOL_SOCKET, SO_REUSEADDR, &i, sizeof(i));
}

YuriNetSocket::YuriNetSocket(int sock_raw):
socket_(sock_raw)
{
}
YuriNetSocket::~YuriNetSocket() noexcept
{
	::close(socket_);
}
bool YuriNetSocket::ready_to_send()
{
	pollfd fds = {socket_, POLLOUT, 0};
	::poll(&fds, 1, 0);
	return (fds.revents & POLLOUT);
}
bool YuriNetSocket::data_available()
{
	return wait_for_data(0_ms);
}
bool YuriNetSocket::wait_for_data(duration_t duration)
{
	pollfd fds = {socket_, POLLIN, 0};
	::poll(&fds, 1, static_cast<int>(duration.value/1000));
	return (fds.revents & POLLIN);
}


}
}


