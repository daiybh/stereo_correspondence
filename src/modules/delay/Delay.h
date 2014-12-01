/*!
 * @file 		Delay.h
 * @author 		<Your name>
 * @date 		01.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef DELAY_H_
#define DELAY_H_

#include "yuri/core/thread/IOThread.h"
#include <deque>

namespace yuri {
namespace delay {

class Delay: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Delay(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Delay() noexcept;
private:
	virtual void run();
	virtual bool set_param(const core::Parameter& param);

	struct frame_time_t {
		core::pFrame frame;
		timestamp_t timestamp;
	};

	duration_t delay_;
	std::deque<frame_time_t> frames_;

};

} /* namespace delay */
} /* namespace yuri */
#endif /* DELAY_H_ */
