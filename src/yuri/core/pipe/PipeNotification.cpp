/*!
 * @file 		PipeNotification.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PipeNotification.h"

namespace yuri {
namespace core {

void PipeNotifiable::notify()
{
	variable_.notify_all();
}
void PipeNotifiable::wait_for(duration_t dur)
{
	lock_t lock(var_mutex_);
	variable_.wait_for(lock, std::chrono::microseconds(dur));
}

}
}

