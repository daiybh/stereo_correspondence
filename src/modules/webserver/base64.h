/*
 * base64.h
 *
 *  Created on: 8. 12. 2014
 *      Author: neneko
 */

#ifndef SRC_MODULES_WEBSERVER_BASE64_H_
#define SRC_MODULES_WEBSERVER_BASE64_H_

#include <string>

namespace yuri {
namespace webserver {
namespace base64 {
std::string encode(std::string);
std::string decode(const std::string&);
}
}
}



#endif /* SRC_MODULES_WEBSERVER_BASE64_H_ */
