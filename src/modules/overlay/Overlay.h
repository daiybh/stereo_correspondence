/*!
 * @file 		Overlay.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.05.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef OVERLAY_H_
#define OVERLAY_H_

//#include "yuri/core/thread/MultiIOFilter.h"
#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace overlay {

class Overlay: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Overlay(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Overlay() noexcept;
	template<bool rewrite, class kernel>
	core::pRawVideoFrame combine(core::pRawVideoFrame frame_0, const core::pRawVideoFrame& frame_1);
private:

	//virtual bool step();
//	virtual std::vector<core::pFrame> do_single_step(const std::vector<core::pFrame>&);
	virtual std::vector<core::pFrame> do_special_step(param_type) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
//	core::pBasicFrame frame_0;
//	core::pBasicFrame frame_1;
	ssize_t x_;
	ssize_t y_;
};

} /* namespace overlay */
} /* namespace yuri */
#endif /* OVERLAY_H_ */
