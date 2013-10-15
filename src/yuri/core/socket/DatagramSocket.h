/*
 * DatagramSocket.h
 *
 *  Created on: 14.10.2013
 *      Author: neneko
 */

#ifndef DATAGRAMSOCKET_H_
#define DATAGRAMSOCKET_H_
#include "socket_errors.h"
#include "yuri/log/Log.h"
#include "yuri/core/utils/time_types.h"
#include <string>
#include <array>
#include <vector>
namespace yuri {
namespace core {
namespace socket {

typedef uint16_t port_t;

class DatagramSocket {
public:

	DatagramSocket(const log::Log& log_, const std::string& url = std::string());
	/*!
	 * Binds socket to local port (if the underlying socket supports this).
	 * @param url Address to bind localy. Exact meaning is implementation dependent, may contain port specification
	 * @param port Port number to bind to. If specified, it should have higher priority than any port specified in @em url
	 * @return false if an error occured, true otherwise
	 */
	bool bind(const std::string& url, port_t port);

	/*!
	 * Checks whether there are any data waiting to be read
	 * @return true if subsequent call to receive_datagram() will succeed.
	 */
	bool data_available();

	bool wait_for_data(duration_t duration);

	/*!
	 * Checks whether the underlying socket is ready to send a packet;
	 * @return
	 */
	bool ready_to_send();

	size_t send_datagram(const uint8_t* data, size_t size);

	template<typename T>
	size_t send_datagram(const T* data, size_t size);
	template<typename T>
	size_t send_datagram(const std::vector<T>& data);
	template<typename T, size_t N>
	size_t send_datagram(const std::array<T, N>& data);

	/*!
	 * Receives a single datagram from socket
	 * @param data Pointer to location to store the data
	 * @param size size of the buffer
	 * @return number of bytes received.
	 */
	size_t receive_datagram(uint8_t* data, size_t size);

	template<typename T>
	size_t receive_datagram(T* data, size_t size);
	template<typename T>
	size_t receive_datagram(std::vector<T>& data);
	template<typename T, size_t N>
	size_t receive_datagram(std::array<T, N>& data);


protected:
	log::Log	log;
private:

	virtual size_t do_send_datagram(const uint8_t* data, size_t size) = 0;
	virtual size_t do_receive_datagram(uint8_t* data, size_t size) = 0;
	virtual bool do_bind(const std::string& url, port_t port) = 0;
	virtual bool do_data_available() = 0;
	virtual bool do_ready_to_send() = 0;
	virtual bool do_wait_for_data(duration_t duration) = 0;
};


template<typename T>
size_t DatagramSocket::send_datagram(const T* data, size_t size)
{
	return send_datagram(reinterpret_cast<const uint8_t*>(data), size * sizeof(T)) / sizeof(T);
}

template<typename T>
size_t DatagramSocket::send_datagram(const std::vector<T>& data)
{
	return send_datagram(reinterpret_cast<const uint8_t*>(&data[0]), data.size() * sizeof(T)) / sizeof(T);
}

template<typename T, size_t N>
size_t DatagramSocket::send_datagram(const std::array<T,N>& data)
{
	return send_datagram(reinterpret_cast<const uint8_t*>(data), N * sizeof(T)) / sizeof(T);
}

template<typename T>
size_t DatagramSocket::receive_datagram(T* data, size_t size)
{
	return receive_datagram(reinterpret_cast<uint8_t*>(data), size * sizeof(T)) / sizeof(T);
}
template<typename T>
size_t DatagramSocket::receive_datagram(std::vector<T>& data)
{
	return receive_datagram(reinterpret_cast<uint8_t*>(&data[0]), data.size() * sizeof(T)) / sizeof(T);
}
template<typename T, size_t N>
size_t DatagramSocket::receive_datagram(std::array<T, N>& data)
{
	return receive_datagram(reinterpret_cast<uint8_t*>(&data[0]), N * sizeof(T)) / sizeof(T);
}

}
}
}



#endif /* DATAGRAMSOCKET_H_ */
