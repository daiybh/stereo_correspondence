/*
 * StreamSocket.cpp
 *
 *  Created on: 27.10.2013
 *      Author: neneko
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



