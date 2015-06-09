/*!
 * @file 		wall_time.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		2. 4. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */
#include "wall_time.h"
#include "yuri/core/utils/new_types.h"
#include <chrono>
namespace yuri {
namespace core {
namespace utils {
namespace {
std::mutex wall_clock_mutex;
const auto startup_time= std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

std::tm get_current_local_time()
{
	auto now = std::chrono::system_clock::now();
	auto tmt = std::chrono::system_clock::to_time_t(now);
	// We have to have a lock around lock std::localtime, as it's not thread safe
	// and returns a pointer to an internal stucture.
	lock_t _(wall_clock_mutex);
	auto t = std::localtime(&tmt);
	std::tm out_time = *t;
	return out_time;
}

std::tm get_startup_local_time()
{
	lock_t _(wall_clock_mutex);
	auto t = std::localtime(&startup_time);
	std::tm out_time = *t;
	return out_time;
}

}
}
}


