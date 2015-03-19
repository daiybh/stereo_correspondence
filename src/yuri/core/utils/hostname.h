/*!
 * @file 		hostname.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		19. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_HOSTNAME_H_
#define SRC_YURI_CORE_UTILS_HOSTNAME_H_

#include "platform.h"

#ifdef YURI_POSIX
#include <unistd.h>
#include <sys/utsname.h>
#else
#endif
namespace yuri {
namespace core {
namespace utils {

std::string get_hostname()
{
#ifdef YURI_POSIX
	std::array<char, 255> name;
	gethostname(&name[0], sizeof(name));
	return std::string(&name[0]);
#else
	// Unsupported platform
	return {};
#endif
}

std::string get_domain()
{
#ifdef YURI_POSIX
	std::array<char, 255> name;
	getdomainname(&name[0], sizeof(name));
	return std::string(&name[0]);
#else
	// Unsupported platform
	return {};
#endif
}


std::string get_sysname()
{
#ifdef YURI_POSIX
	utsname uts;
	uname(&uts);
	return std::string(uts.sysname);
#elif define(YURI_WIN)
	return "Windows";
#else
	// Unsupported platform
	return {};
#endif
}

std::string get_sysver()
{
#ifdef YURI_POSIX
	utsname uts;
	uname(&uts);
	return std::string(uts.sysname) + "-" + std::string(uts.release);
#elif define(YURI_WIN)
	return "Windows";
#else
	// Unsupported platform
	return {};
#endif
}

}
}

}




#endif /* SRC_YURI_CORE_UTILS_HOSTNAME_H_ */
