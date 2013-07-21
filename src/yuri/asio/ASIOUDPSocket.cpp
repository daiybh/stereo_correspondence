/*!
 * @file 		ASIOUDPSocket.cpp
 * @author 		Zdenek Travnicek
 * @date 		27.10.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "ASIOUDPSocket.h"
#include "yuri/exception/InitializationFailed.h"
#include "boost/lexical_cast.hpp"
namespace yuri
{

namespace asio
{

ASIOUDPSocket::ASIOUDPSocket(log::Log &_log, core::pwThreadBase parent,yuri::ushort_t port):
		core::SocketBase(_log,parent),socket(0)
{
	log.set_label("[ASIO UDP] ");
	try {
		socket=new udp::socket(io_service);
		socket->open(udp::v4());
		socket->set_option(boost::asio::socket_base::reuse_address(true));
		socket->bind(udp::endpoint(udp::v4(), port));
	}
	catch (std::exception &e) {
		log[log::error] << "EEEEEERRRRRRRRRROOOOOOOOOORRRRRRRRRRRRRRRRRRR!!\n";
		throw exception::InitializationFailed(std::string("Failed to initialize UDP socket: ")+e.what());
	}
	try {boost::asio::socket_base::send_buffer_size opt;
		boost::asio::socket_base::send_buffer_size opt2(10485760);
		boost::asio::socket_base::receive_buffer_size opt3(10485760);
		socket->get_option(opt);
		log[log::debug] << "Send size: " << opt.value() << "\n";
		socket->set_option(opt2);
		socket->set_option(opt3);
		socket->get_option(opt);
		log[log::debug] << "Send size: " << opt.value() << "\n";
		log[log::debug] << "Receive size: " << opt3.value() << "\n";

	}
	catch (std::exception &e) {
		log[log::warning] << "Failed to increase buffer sizes....!!"<<"\n";
		//throw InitializationFailed(std::string("Failed to initialize UDP socket: ")+e.what());
	}
}

ASIOUDPSocket::~ASIOUDPSocket()
{
}

yuri::size_t ASIOUDPSocket::read(yuri::ubyte_t * data,yuri::size_t size)
{
	yuri::size_t recvd =
		socket->receive_from(boost::asio::buffer(data,size), remote_endpoint, 0);
	//log[log::debug] << "Received data from " << remote_endpoint.address().to_string()
	//	<< "\n";
	return recvd;

}
yuri::size_t ASIOUDPSocket::write(const yuri::ubyte_t * data,yuri::size_t size)
{
	return socket->send_to(boost::asio::buffer(data,size), remote_endpoint);
}

bool ASIOUDPSocket::bind_local(std::string addr,yuri::ushort_t port)
{
	if (addr=="") return bind_local(port);
	return true;
}
bool ASIOUDPSocket::bind_local(yuri::ushort_t /*port*/)
{
	return true;
}

void ASIOUDPSocket::set_port(yuri::ushort_t port)
{
	this->port=port;
	//socket=udp::socket(io_service, udp::endpoint(udp::v4(), port));
}

bool ASIOUDPSocket::data_available()
{
	return socket->available();
}

int ASIOUDPSocket::get_fd()
{
	return (int) socket->native();
}

bool ASIOUDPSocket::set_endpoint(std::string address, yuri::size_t port)
{
	udp::resolver resolver(io_service);
	udp::resolver::query query(udp::v4(), address, boost::lexical_cast<std::string>(port));
	log[log::debug] << "Resolving " << address << "\n";
	remote_endpoint = *resolver.resolve(query);
	log[log::debug] << "Resolved to " << remote_endpoint.address().to_string() << "\n";
	if (remote_endpoint.address().is_v4()) {

		boost::asio::ip::address_v4 v4addr = boost::asio::ip::address_v4::from_string(remote_endpoint.address().to_string());
		if (v4addr.is_multicast()) {
			boost::asio::ip::multicast::join_group option(remote_endpoint.address());
			socket->set_option(option);
		}
	}
	return true;
}

void ASIOUDPSocket::disable_checksums(bool disable_checksums)
{
//ifdef __CYGWIN32__ || __WIN32__
#ifdef __linux__
	socket_no_check sck(disable_checksums);
	socket->set_option(sck);		
#endif
}

}

}
// End of File
