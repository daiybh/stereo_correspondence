/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@cesnet.cz>
 * @date		01.12.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD or GPL License,
 * 				see /doc/LICENSE.txt for details
 *
 */

#include "WebServer.h"
#include "WebStaticResource.h"
#include "WebImageResource.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace webserver {



MODULE_REGISTRATION_BEGIN("webserver")
		REGISTER_IOTHREAD("webserver",WebServer)
		REGISTER_IOTHREAD("web_static",WebStaticResource)
		REGISTER_IOTHREAD("web_image",WebImageResource)

MODULE_REGISTRATION_END()

}
}
