/*!
 * @file 		Timer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		8.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
