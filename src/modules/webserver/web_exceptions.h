/*!
 * @file 		web_exceptions.h
 * @author 		Zdenek Travnicek <travnicek@cesnet.cz>
 * @date		14.12.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD or GPL License,
 * 				see /doc/LICENSE.txt for details
 *
 */

#ifndef SRC_MODULES_WEBSERVER_WEB_EXCEPTIONS_H_
#define SRC_MODULES_WEBSERVER_WEB_EXCEPTIONS_H_
#include <stdexcept>
#include <string>
#include "common_types.h"
namespace yuri {
namespace webserver {

class redirect_to: public std::runtime_error
{
public:
	redirect_to(std::string location, http_code code=http_code::found)
		:runtime_error("Redirecting to "+location),location_(location),code_(code) {}

	const std::string& get_location() { return location_; }
	http_code get_code() { return code_; }
private:
	const std::string location_;
	const http_code code_;
};

}
}





#endif /* SRC_MODULES_WEBSERVER_WEB_EXCEPTIONS_H_ */
