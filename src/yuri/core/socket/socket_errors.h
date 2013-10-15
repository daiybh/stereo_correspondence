/*
 * socket_errors.h
 *
 *  Created on: 15.10.2013
 *      Author: neneko
 */

#ifndef SOCKET_ERRORS_H_
#define SOCKET_ERRORS_H_

#include "yuri/core/utils/new_types.h"
#include <string>
namespace yuri {
namespace core {
namespace socket {

class host_not_found: public std::runtime_error
{
	host_not_found(const std::string& hostname):std::runtime_error("Host "+hostname+" not found"){}
};

}
}
}

#endif /* SOCKET_ERRORS_H_ */
