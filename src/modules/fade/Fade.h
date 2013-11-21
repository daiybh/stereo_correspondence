/*!
 * @file 		Fade.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		13.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FADE_H_
#define FADE_H_

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace fade {

class Fade: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>,
			public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								Fade(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~Fade() noexcept;
private:

	virtual bool 				set_param(const core::Parameter& param) override;
	virtual std::vector<core::pFrame>
								do_special_step(const std::tuple<core::pRawVideoFrame, core::pRawVideoFrame>& frames) override;
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	double						transition_;
};

} /* namespace fade */
} /* namespace yuri */
#endif /* FADE_H_ */
