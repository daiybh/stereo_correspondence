/*!
 * @file 		UnixDatagramSocket.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UNIXDATAGRAMSOCKET_H_
#define UNIXDATAGRAMSOCKET_H_

#include "YuriDatagram.h"

namespace yuri {
namespace network {

class UnixDatagramSocket: public YuriDatagram
{
public:
	UnixDatagramSocket(const log::Log &log_, const std::string&);
	virtual ~UnixDatagramSocket() noexcept;
private:
	virtual bool do_bind(const std::string& url, uint16_t port) override;
	virtual bool do_connect(const std::string& address, uint16_t port) override;

};


}
}




#endif /* UNIXDATAGRAMSOCKET_H_ */
