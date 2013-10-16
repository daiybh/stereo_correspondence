/*!
 * @file 		UVUdpSocket.h
 * @author 		<Your name>
 * @date 		15.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVUDPSOCKET_H_
#define UVUDPSOCKET_H_

#include "yuri/core/socket/DatagramSocket.h"
#define HAVE_CONFIG_H 1
#include "rtp/net_udp.h"
#undef HAVE_CONFIG_H

// OMG, there are macros max/min define in uv
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace yuri {
namespace uv_udp {

class UVUdpSocket: public core::socket::DatagramSocket
{
public:

	UVUdpSocket(const log::Log &log_, const std::string& url);
	virtual ~UVUdpSocket() noexcept;
private:
	virtual size_t do_send_datagram(const uint8_t* data, size_t size) override;
	virtual size_t do_receive_datagram(uint8_t* data, size_t size) override;
	virtual bool do_bind(const std::string& url, core::socket::port_t port) override;
	virtual bool do_data_available() override;
	virtual bool do_ready_to_send() override;
	virtual bool do_wait_for_data(duration_t duration) override;

	unique_ptr<socket_udp,function<void(socket_udp*)>> socket_;
};

} /* namespace uv_udp */
} /* namespace yuri */
#endif /* UVUDPSOCKET_H_ */
