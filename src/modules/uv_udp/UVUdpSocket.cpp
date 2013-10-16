/*!
 * @file 		UVUdpSocket.cpp
 * @author 		<Your name>
 * @date		15.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVUdpSocket.h"
//include "yuri/core/Module.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"


namespace yuri {
namespace uv_udp {



MODULE_REGISTRATION_BEGIN("uv_udp")
		REGISTER_DATAGRAM_SOCKET("uv_udp",UVUdpSocket)
MODULE_REGISTRATION_END()


UVUdpSocket::UVUdpSocket(const log::Log &log_, const std::string& url):
core::socket::DatagramSocket(log_,url),socket_(nullptr,[](socket_udp*s){if(s){udp_exit(s);}})
{
}

UVUdpSocket::~UVUdpSocket() noexcept
{
}


size_t UVUdpSocket::do_send_datagram(const uint8_t* data, size_t size)
{
	// !TODO Should be fixed upstream for const correctness) and const_cast removed!
	if (socket_.get()==nullptr) throw core::socket::socket_not_connected();
	return udp_send(socket_.get(),
			const_cast<char*>(reinterpret_cast<const char*>(data)),
			static_cast<int>(size));
}

size_t UVUdpSocket::do_receive_datagram(uint8_t* data, size_t size) {
	if (socket_.get()==nullptr) throw core::socket::socket_not_connected();
	return udp_recv(socket_.get(),
			reinterpret_cast<char*>(data),
			static_cast<int>(size));
}

bool UVUdpSocket::do_bind(const std::string& url, core::socket::port_t port) {
	socket_.reset(udp_init(/*url.c_str()*/nullptr,port,0,255,false));
	return socket_.get();
}

bool UVUdpSocket::do_data_available() {
	if (socket_.get()==nullptr) throw core::socket::socket_not_connected();
	char c;
	return udp_peek(socket_.get(),&c, 1)!=0;
}

bool UVUdpSocket::do_ready_to_send() {
	return true; // whatever ;)
}

bool UVUdpSocket::do_wait_for_data(duration_t /*duration*/) {
	return true; // TODO: Implement this....
}



} /* namespace uv_udp */
} /* namespace yuri */
