/*!
 * @file 		YuriDatagram.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriDatagram.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>



namespace yuri {
namespace network {


YuriDatagram::YuriDatagram(const log::Log &log_, const std::string&, int domain):
core::socket::DatagramSocket(log_),socket_(domain, SOCK_DGRAM, 0)
{
}

YuriDatagram::~YuriDatagram() noexcept
{
}



size_t YuriDatagram::do_send_datagram(const uint8_t* data, size_t data_size)
{
	ssize_t wrote = ::send(get_socket(), data, data_size, 0);
	return (wrote>0)?wrote:0;
}
size_t YuriDatagram::do_receive_datagram(uint8_t* data, size_t size)
{
	ssize_t read = ::recv(get_socket(), data, size, MSG_DONTWAIT);
	return (read>0)?read:0;
}


bool YuriDatagram::do_data_available()
{
	return socket_.data_available();
}
bool YuriDatagram::do_wait_for_data(duration_t duration)
{
	return socket_.wait_for_data(duration);
}


bool YuriDatagram::do_ready_to_send() {
	return socket_.ready_to_send();
}

} /* namespace yuri_tcp */
} /* namespace yuri */
