/*
 * DatagramSocket.cpp
 *
 *  Created on: 14.10.2013
 *      Author: neneko
 */

#include "DatagramSocket.h"


namespace yuri {
namespace core {
namespace socket {


size_t DatagramSocket::send_datagram(const uint8_t* data, size_t size) {
	return do_send_datagram(data, size);
}
size_t DatagramSocket::receive_datagram(uint8_t* data, size_t size) {
	return do_receive_datagram(data, size);
}
bool DatagramSocket::bind(const std::string& url, port_t port) {
	return do_bind(url, port);
}
bool DatagramSocket::data_available() {
	return do_data_available();
}
bool DatagramSocket::ready_to_send() {
	return do_ready_to_send();
}
bool DatagramSocket::wait_for_data(duration_t duration) {
	return do_wait_for_data(duration);
}


}
}
}


