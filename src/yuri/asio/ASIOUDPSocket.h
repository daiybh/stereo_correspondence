/*!
 * @file 		ASIOUDPSocket.h
 * @author 		Zdenek Travnicek
 * @date 		27.10.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef ASIOUDPSOCKET_H_
#define ASIOUDPSOCKET_H_

#include "boost/asio.hpp"
#include "yuri/core/SocketBase.h"

namespace yuri
{

namespace asio
{
using boost::asio::ip::udp;
#ifdef YURI_LINUX
typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_NO_CHECK> socket_no_check;
#else
#endif

class ASIOUDPSocket : public core::SocketBase
{
public:
	ASIOUDPSocket(log::Log &_log, core::pwThreadBase parent,yuri::ushort_t port);
	virtual ~ASIOUDPSocket();
	virtual yuri::size_t read(yuri::ubyte_t * data,yuri::size_t size);
	virtual yuri::size_t write(const yuri::ubyte_t * data,yuri::size_t size);
	bool bind_local(std::string addr, yuri::ushort_t port) ;
	bool bind_local(yuri::ushort_t port);
	void set_port(yuri::ushort_t port);
	virtual bool data_available();
	virtual int get_fd();
	bool set_endpoint(std::string address, yuri::size_t port);
	void disable_checksums(bool disable_checksums = true);
protected:
	boost::asio::io_service io_service;
	yuri::ushort_t port;
	udp::socket *socket;
	udp::endpoint remote_endpoint;
};

}

}

#endif /*ASIOUDPSOCKET_H_*/
