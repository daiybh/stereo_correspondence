/*!
 * @file 		PipeNotification.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		1.12.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PipeNotification.h"

namespace yuri {
namespace core {

void PipeNotifiable::notify()
{
	{
		lock_t lock(var_mutex_);
		pending_notification_=true;
	}
	variable_.notify_all();
}
void PipeNotifiable::wait_for(duration_t dur)
{
	lock_t lock(var_mutex_);
	if (pending_notification_) {
		pending_notification_ = false;
		return;
	}
	const auto status = variable_.wait_for(lock, std::chrono::microseconds(dur));
	if (status == std::cv_status::no_timeout) {
		pending_notification_ = false;
	}
}

}
}

