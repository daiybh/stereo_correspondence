/*
 * PipeNotification.cpp
 *
 *  Created on: 9.9.2013
 *      Author: neneko
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

