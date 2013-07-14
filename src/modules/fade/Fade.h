/*!
 * @file 		Fade.h
 * @author 		<Your name>
 * @date 		13.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef FADE_H_
#define FADE_H_

#include "yuri/core/BasicIOFilter.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace fade {

class Fade: public core::BasicMultiIOFilter, public event::BasicEventConsumer
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters 	configure();
	virtual 					~Fade();
private:
								Fade(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool 				set_param(const core::Parameter& param);
	virtual std::vector<core::pBasicFrame>
								do_single_step(const std::vector<core::pBasicFrame>& frames);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);

	double						transition_;
};

} /* namespace fade */
} /* namespace yuri */
#endif /* FADE_H_ */
