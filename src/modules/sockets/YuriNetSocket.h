/*!
 * @file 		YuriNetSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURINETSOCKET_H_
#define YURINETSOCKET_H_
#include "yuri/core/utils/time_types.h"
namespace yuri {
namespace network {

class YuriNetSocket {
public:
	YuriNetSocket(int domain, int type, int proto = 0);
	YuriNetSocket(int sock_raw);
	~YuriNetSocket() noexcept;
	YuriNetSocket(const YuriNetSocket&) = delete;
	YuriNetSocket& operator=(const YuriNetSocket&) = delete;


	int get_socket() const { return socket_; }


	bool ready_to_send();
	bool data_available();
	bool wait_for_data(duration_t duration);
private:
	int socket_;


};

}
}



#endif /* YURINETSOCKET_H_ */
