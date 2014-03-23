/*!
 * @file 		StreamSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef STREAMSOCKET_H_
#define STREAMSOCKET_H_

#include "socket_errors.h"
#include "yuri/log/Log.h"
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/time_types.h"
#include "yuri/core/utils/uvector.h"
#include <string>
#include <vector>

namespace yuri {
namespace core {
namespace socket {
class StreamSocket;
typedef shared_ptr<StreamSocket> pStreamSocket;
class StreamSocket
{
public:
	StreamSocket(const log::Log& log_);
	virtual ~StreamSocket() noexcept;

	bool bind(const std::string& url, uint16_t port);
	bool connect(const std::string& address, uint16_t port);

	size_t send_data(const uint8_t* data, size_t data_size);

	template<class T>
	size_t send_data(const T* data, size_t data_size)
	{
		return send_data(reinterpret_cast<const uint8_t*>(data), data_size * sizeof(T))/sizeof(T);
	}
	template<class T>
	size_t send_data(const std::vector<T>& data)
	{
		return send_data(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T))/sizeof(T);
	}
	template<class T, size_t N>
	size_t send_data(const std::array<T, N>& data)
	{
		return send_data(reinterpret_cast<const uint8_t*>(data.data()), N * sizeof(T))/sizeof(T);
	}
	template<class Char>
	size_t send_data(const std::basic_string<Char>& data)
	{
		return send_data(data.data(), data.size());
	}

	size_t receive_data(uint8_t* data, size_t size);


	template<class T>
	size_t receive_data(std::vector<T>& data)
	{
		return receive_data(reinterpret_cast<uint8_t*>(data.data()), data.size()*sizeof(T))/sizeof(T);
	}
	template<class T>
	size_t receive_data(uvector<T>& data)
	{
		return receive_data(reinterpret_cast<uint8_t*>(data.data()), data.size()*sizeof(T))/sizeof(T);
	}
	template<class T, size_t N>
	size_t receive_data(std::array<T, N>& data)
	{
		return receive_data(reinterpret_cast<uint8_t*>(data.data()), N*sizeof(T))/sizeof(T);
	}
	/*!
	 * Checks whether there are any data waiting to be read
	 * @return true if subsequent call to receive_datagram() will succeed.
	 */
	bool data_available();

	bool wait_for_data(duration_t duration);
protected:
	log::Log		log;
private:
	virtual size_t do_send_data(const uint8_t* data, size_t data_size) = 0;
	virtual size_t do_receive_data(uint8_t* data, size_t size) = 0;
	virtual bool do_bind(const std::string& url, uint16_t port) = 0;
	virtual bool do_connect(const std::string& address, uint16_t port) = 0;
	virtual bool do_data_available() = 0;
	virtual bool do_wait_for_data(duration_t duration) = 0;


};

}
}
}


#endif /* STREAMSOCKET_H_ */
