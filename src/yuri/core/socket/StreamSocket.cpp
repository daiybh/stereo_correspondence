/*!
 * @file 		StreamSocket.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "StreamSocket.h"


namespace yuri {
namespace core {
namespace socket {

StreamSocket::StreamSocket(const log::Log& log_)
:log(log_)
{

}

StreamSocket::~StreamSocket() noexcept
{
}

size_t StreamSocket::send_data(const uint8_t* data, size_t data_size)
{
	return do_send_data(data, data_size);
}


size_t StreamSocket::receive_data(uint8_t* data, size_t size)
{
	return do_receive_data(data, size);
}


bool StreamSocket::bind(const std::string& url, uint16_t port)
{
	return do_bind(url, port);
}
bool StreamSocket::connect(const std::string& address, uint16_t port)
{
	return do_connect(address, port);
}
bool StreamSocket::listen()
{
	return do_listen();
}
pStreamSocket StreamSocket::accept()
{
	return do_accept();
}

bool StreamSocket::data_available()
{
	return do_data_available();
}
bool StreamSocket::wait_for_data(duration_t duration)
{
	return do_wait_for_data(duration);
}

}
}
}



