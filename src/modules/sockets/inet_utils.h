/*
 * YuriInetSocket.h
 *
 *  Created on: 19. 2. 2015
 *      Author: neneko
 */

#ifndef YURIINETSOCKET_H_
#define YURIINETSOCKET_H_
#include "YuriNetSocket.h"
namespace yuri{
namespace network {
namespace inet {

	bool bind(YuriNetSocket& socket, const std::string& url, uint16_t port);
	bool connect(YuriNetSocket& socket, const std::string& address, uint16_t port);

}

}
}

#endif /* YURIINETSOCKET_H_ */
