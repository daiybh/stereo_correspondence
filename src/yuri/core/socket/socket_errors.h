/*!
 * @file 		socket_errors.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
public:
	host_not_found(const std::string& hostname):std::runtime_error("Host "+hostname+" not found"){}
};
class socket_not_connected: public std::runtime_error
{
public:
	socket_not_connected():std::runtime_error("Socket is not connected"){}
};

}
}
}

#endif /* SOCKET_ERRORS_H_ */
