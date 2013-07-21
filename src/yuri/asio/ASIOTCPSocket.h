/*!
 * @file 		ASIOTCPSocket.cpp
 * @author 		Zdenek Travnicek
 * @date 		20.12.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef ASIOTCPSOCKET_H_
#define ASIOTCPSOCKET_H_
#include "boost/asio.hpp"
#include "yuri/core/SocketBase.h"
#include "yuri/exception/InitializationFailed.h"
namespace yuri
{

namespace asio
{
using boost::asio::ip::tcp;

//typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_NO_CHECK> socket_no_check;

class ASIOTCPSocket : public core::SocketBase {
public:
	ASIOTCPSocket(log::Log &_log, core::pwThreadBase parent);
	virtual ~ASIOTCPSocket();
	bool connect(std::string address, yuri::ushort_t port);
	virtual yuri::size_t read(yuri::ubyte_t * data,yuri::size_t size);
	virtual yuri::size_t write(const yuri::ubyte_t * data,yuri::size_t size);
	virtual int get_fd();
	virtual yuri::size_t available();
protected:
	boost::asio::io_service io_service;
	boost::shared_ptr<tcp::socket> socket;
	tcp::endpoint remote_endpoint;

};

}

}

#endif /* ASIOTCPSOCKET_H_ */
