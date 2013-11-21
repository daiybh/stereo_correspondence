/*!
 * @file 		ASIOTCPSocket.cpp
 * @author 		Zdenek Travnicek
 * @date 		20.12.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2011 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "ASIOTCPSocket.h"
#include "boost/lexical_cast.hpp"
namespace yuri {

namespace asio {

ASIOTCPSocket::ASIOTCPSocket(log::Log &_log, core::pwThreadBase parent):core::SocketBase(_log,parent)
{
	try {
		socket.reset(new tcp::socket(io_service));
	}
	catch (...) {
		throw exception::InitializationFailed("Failed to iniailize ASIOTCPSocket");
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
	return false;
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
yuri::size_t ASIOTCPSocket::write(const yuri::ubyte_t * data,yuri::size_t size)
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
