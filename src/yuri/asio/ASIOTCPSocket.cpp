/*
 * ASIOTCPSocket.cpp
 *
 *  Created on: Dec 20, 2011
 *      Author: neneko
 */

#include "ASIOTCPSocket.h"

namespace yuri {

namespace io {

ASIOTCPSocket::ASIOTCPSocket(Log &_log,pThreadBase parent):SocketBase(_log,parent)
{
	try {
		socket.reset(new tcp::socket(io_service));
	}
	catch (...) {
		throw InitializationFailed("Failed to iniailize ASIOTCPSocket");
	}
}

ASIOTCPSocket::~ASIOTCPSocket()
{

}

bool ASIOTCPSocket::connect(std::string address, yuri::ushort_t port)
{
	tcp::resolver resolver(io_service);
	tcp::resolver::query query(tcp::v4(), address, boost::lexical_cast<std::string>(port));
	tcp::resolver::iterator iterator = resolver.resolve(query);
	while (true) {
		try {
			socket->connect(*iterator);
			return true;
		}
		catch (boost::system::system_error &e) {
			if (e.code() == boost::system::errc::interrupted) continue;
			throw (e);
		}
	}
}

yuri::size_t ASIOTCPSocket::read(yuri::ubyte_t * data,yuri::size_t size)
{
	yuri::size_t recvd;
	while (1) {
		try {
			recvd =	socket->receive(boost::asio::buffer(data,size));
		}
		catch (boost::system::system_error &e) {
			if (e.code() == boost::system::errc::interrupted) continue;
			throw (e);
		}
		break;
	}
	return recvd;

}
yuri::size_t ASIOTCPSocket::write(yuri::ubyte_t * data,yuri::size_t size)
{
	return socket->send(boost::asio::buffer(data,size));
}
int ASIOTCPSocket::get_fd()
{
	return (int) socket->native();
}
yuri::size_t ASIOTCPSocket::available()
{
	return socket->available();
}
}

}
