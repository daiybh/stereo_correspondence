/*!
 * @file 		hostname.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		19. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */



#include "hostname.h"
#include "platform.h"

#ifdef YURI_POSIX
#include <array>
#include <unistd.h>
#include <sys/utsname.h>
#elif defined(YURI_WIN)
#include <array>
#include <WinSock2.h>
#include <Windows.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#endif
namespace yuri {
namespace core {
namespace utils {

#if defined(YURI_WIN)
	namespace {
		void init_wsa() {
			static bool initialized = false;
			if (!initialized) {
				WORD req = MAKEWORD(2, 2);
				WSADATA data;
				WSAStartup(req, &data);
				initialized = true;
			}
		}
	}
#endif


std::string get_hostname()
{
#ifdef YURI_POSIX
	std::array<char, 255> name;
	gethostname(&name[0], sizeof(name));
	return std::string(&name[0]);
#elif defined(YURI_WIN)
	init_wsa();
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
#elif defined(YURI_WIN)
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
#elif defined(YURI_WIN)
	OSVERSIONINFO ver;
	ZeroMemory(&ver, sizeof(OSVERSIONINFO));
	ver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&ver);
	auto v = std::string(ver.szCSDVersion);
	return "Windows-"+std::to_string(ver.dwMajorVersion)+"." + std::to_string(ver.dwMinorVersion) + v;
#else
	// Unsupported platform
	return {};
#endif
}

}
}

}

