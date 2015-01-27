/*!
 * @file 		DatagramSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		14.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
class DatagramSocket;
typedef std::shared_ptr<DatagramSocket> pDatagramSocket;
class DatagramSocket {
public:

	EXPORT DatagramSocket(const log::Log& log_, const std::string& url = std::string());
	EXPORT virtual ~DatagramSocket() noexcept;
	/*!
	 * Binds socket to local port (if the underlying socket supports this).
	 * @param url Address to bind localy. Exact meaning is implementation dependent, may contain port specification
	 * @param port Port number to bind to. If specified, it should have higher priority than any port specified in @em url
	 * @return false if an error occured, true otherwise
	 */
	EXPORT bool bind(const std::string& url, port_t port);

	/*!
	 * Connects socket to remote host (if the underlying socket supports this).
	 * @param url Address of remote host. Exact meaning is implementation dependent, may contain port specification
	 * @param port Port number to bind to. If specified, it should have higher priority than any port specified in @em url
	 * @return false if an error occured, true otherwise
	 */
	EXPORT	bool connect(const std::string& url, port_t port);
	/*!
	 * Checks whether there are any data waiting to be read. Function returns immediately.
	 * @return true if subsequent call to receive_datagram() will succeed.
	 */
	EXPORT bool data_available();

	/*!
	 * Waits for data to become available, for at most @em duration.
	 * @param duration Maximum time to wait for data
	 * @return true if there are data available for read, false if timeout occurred before data become available.
	 */
	EXPORT bool wait_for_data(duration_t duration);

	/*!
	 * Checks whether the underlying socket is ready to send a packet;
	 * @return
	 */
	EXPORT bool ready_to_send();

	/*!
	 * Sends datagram
	 * @param data Pinter to data to send
	 * @param size size of datagram in bytes
	 * @return number of bytes really sent
	 */
	EXPORT size_t send_datagram(const uint8_t* data, size_t size);

	/*!
	 * Convenience wrapper for sending datagrams with different underlying type
	 * @param data Pointer to beginning of data
	 * @param size Number of elements of type @em T
	 * @return number of elements really sent
	 */
	template<typename T>
	size_t send_datagram(const T* data, size_t size);
	/*!
	 * Convenience wrapper for sending data stored in a std::vector (sends whole vector)
	 * @param data Vector to be sent
	 * @return number of elements sent
	 */
	template<typename T>
	size_t send_datagram(const std::vector<T>& data);
	/*!
	 * Convenience wrapper for sending data stored in a std::array (sends whole array)
	 * @param data Array to be sent
	 * @return number of elements sent
	 */
	template<typename T, size_t N>
	size_t send_datagram(const std::array<T, N>& data);
	/*!
	 * Convenience wrapper for sending data stored in a std::basic_string (sends whole string)
	 * @param data String to send
	 * @return number of characters sent
	 */
	template<typename T>
	size_t send_datagram(const std::basic_string<T>& data);

	/*!
	 * Receives a single datagram from socket
	 * @param data Pointer to location to store the data
	 * @param size size of the buffer
	 * @return number of bytes received.
	 */
	EXPORT size_t receive_datagram(uint8_t* data, size_t size);

	/*!
	 * Convenience wrapper, receives datagram into an array with different underlying type
	 * @param data Beginning of data
	 * @param size Number of elements available
	 * @return Number of elements stored
	 */
	template<typename T>
	size_t receive_datagram(T* data, size_t size);
	/*!
	 * Convenience wrapper, receives datagram into a std::vector
	 * Note that the method does NOT modify the length of the vector!
	 *
	 * @param data The vector to write to
	 * @return Number of elements stored
	 */
	template<typename T>
	size_t receive_datagram(std::vector<T>& data);
	/*!
	 * Convenience wrapper, receives datagram into a std::array
	 * @param data The array to write to
	 * @return Number of elements stored
	 */
	template<typename T, size_t N>
	size_t receive_datagram(std::array<T, N>& data);

protected:
	log::Log	log;
private:

	virtual size_t do_send_datagram(const uint8_t* data, size_t size) = 0;
	virtual size_t do_receive_datagram(uint8_t* data, size_t size) = 0;
	virtual bool do_bind(const std::string& url, port_t port) = 0;
	virtual bool do_connect(const std::string& url, port_t port) = 0;
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
size_t DatagramSocket::send_datagram(const std::basic_string<T>& data)
{
	return send_datagram(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T)) / sizeof(T);
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
