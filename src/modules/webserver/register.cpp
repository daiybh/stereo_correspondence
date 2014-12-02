/*
 * register.cpp
 *
 *  Created on: 1. 12. 2014
 *      Author: neneko
 */

#include "WebServer.h"
#include "WebStaticResource.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace webserver {



MODULE_REGISTRATION_BEGIN("webserver")
		REGISTER_IOTHREAD("webserver",WebServer)
		REGISTER_IOTHREAD("web_static",WebStaticResource)

MODULE_REGISTRATION_END()

}
}
