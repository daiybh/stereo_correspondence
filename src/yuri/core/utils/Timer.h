/*
 * timer.h
 *
 *  Created on: 8.9.2013
 *      Author: neneko
 */

#ifndef TIMER_H_
#define TIMER_H_

#include "time_types.h"
namespace yuri {
class Timer
{
public:
	Timer() {reset();}
	~Timer() noexcept {}
	inline void						reset() noexcept {
		last_ = current_time();
	}
	inline duration_t				get_duration() noexcept {
		return duration_t(current_time() - last_);
	}
private:
	detail::time_point				current_time() { return detail::clock_t::now(); }
	detail::time_point				last_;
};


// TODO ???
//class FPSTimer {
//public:
//	FPSTimer(double fps):fps_(fps),frame_count_(0){}
//
//
//
//private:
//	Timer 						timer_;
//	double						fps_;
//	size_t 						frame_count_;
//};

}



#endif /* TIMER_H_ */
