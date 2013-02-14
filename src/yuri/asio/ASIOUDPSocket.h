#ifndef ASIOUDPSOCKET_H_
#define ASIOUDPSOCKET_H_

#include "boost/asio.hpp"
#include "yuri/io/SocketBase.h"
#include "yuri/exception/InitializationFailed.h"
namespace yuri
{

namespace io
{
using boost::asio::ip::udp;
using yuri::exception::InitializationFailed;
#ifdef __linux__
typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_NO_CHECK> socket_no_check;
#else
#endif
class ASIOUDPSocket : public SocketBase
{
public:
	ASIOUDPSocket(Log &_log,pThreadBase parent,yuri::ushort_t port) throw (InitializationFailed);
	virtual ~ASIOUDPSocket();
	virtual yuri::size_t read(yuri::ubyte_t * data,yuri::size_t size);
	virtual yuri::size_t write(yuri::ubyte_t * data,yuri::size_t size);
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
