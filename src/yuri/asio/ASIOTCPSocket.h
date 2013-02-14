/*
 * ASIOTCPSocket.h
 *
 *  Created on: Dec 20, 2011
 *      Author: neneko
 */

#ifndef ASIOTCPSOCKET_H_
#define ASIOTCPSOCKET_H_
#include "boost/asio.hpp"
#include "yuri/io/SocketBase.h"
#include "yuri/exception/InitializationFailed.h"
namespace yuri
{

namespace io
{
using boost::asio::ip::tcp;
using yuri::exception::InitializationFailed;
//typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_NO_CHECK> socket_no_check;

class ASIOTCPSocket : public SocketBase {
public:
	ASIOTCPSocket(Log &_log,pThreadBase parent);
	virtual ~ASIOTCPSocket();
	bool connect(std::string address, yuri::ushort_t port);
	virtual yuri::size_t read(yuri::ubyte_t * data,yuri::size_t size);
	virtual yuri::size_t write(yuri::ubyte_t * data,yuri::size_t size);
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
