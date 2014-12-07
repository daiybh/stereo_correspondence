/*
 * common_types.h
 *
 *  Created on: Dec 7, 2014
 *      Author: neneko
 */

#ifndef SRC_MODULES_WEBSERVER_COMMON_TYPES_H_
#define SRC_MODULES_WEBSERVER_COMMON_TYPES_H_
#include <string>
#include <map>
#include "yuri/core/socket/StreamSocket.h"

namespace yuri {
namespace webserver {
enum class http_code {
	continue_ = 100,
	ok = 200,
	created = 201,
	accepted = 202,
	no_content = 204,
	partial = 206,
	moved = 301,
	found = 302,
	see_other = 303,
	not_modified = 304,
	bad_request = 400,
	unauthorized = 401,
	forbidden = 403,
	not_found = 404,
	gone = 410,
	server_error = 500,
	service_unavailable = 503
};


using parameters_t = std::map<std::string, std::string>;
struct request_t
{
	std::string url;
	parameters_t parameters;
	core::socket::pStreamSocket client;
};

struct response_t
{
	http_code code;
	parameters_t parameters;
	std::string data;
};

}
}




#endif /* SRC_MODULES_WEBSERVER_COMMON_TYPES_H_ */
