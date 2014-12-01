/*!
 * @file 		Delay.cpp
 * @author 		<Your name>
 * @date		01.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Delay.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace delay {


IOTHREAD_GENERATOR(Delay)

MODULE_REGISTRATION_BEGIN("delay")
		REGISTER_IOTHREAD("delay",Delay)
MODULE_REGISTRATION_END()

core::Parameters Delay::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Delay");
	p["delay"]["Delay of frame in seconds"]=5.0;
	return p;
}


Delay::Delay(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("delay"))
{
	IOTHREAD_INIT(parameters)
}

Delay::~Delay() noexcept
{
}

void Delay::run()
{
	while (still_running()) {
		while (auto frame = pop_frame(0)) {
			frames_.push_back({frame,{}});
		}

		duration_t next_wait = get_latency();

		if (!frames_.empty()) {
			timestamp_t current_time;
			bool step_finished = false;
			while (!step_finished && !frames_.empty()) {
				auto& newest =  frames_.front();
				const auto delta = current_time - newest.timestamp;
				if (delta >= delay_) {
					push_frame(0,newest.frame);
					frames_.pop_front();
				} else {
					step_finished = true;
					const auto& wanna_wait = (delay_ - delta)/2.0;
					if (wanna_wait < next_wait) next_wait = wanna_wait;
				}
			}
		}
		if (!pipes_data_available()) {
			wait_for(next_wait);
		}
	}
}

bool Delay::set_param(const core::Parameter& param)
{
	if (param.get_name() == "delay") {
		delay_ = 1_s * param.get<double>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace delay */
} /* namespace yuri */
