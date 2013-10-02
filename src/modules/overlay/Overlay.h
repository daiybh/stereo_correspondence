/*!
 * @file 		Overlay.h
 * @author 		<Your name>
 * @date 		27.05.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
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
	virtual ~Overlay();
	template<class kernel>
	core::pRawVideoFrame combine(const core::pRawVideoFrame& frame_0, const core::pRawVideoFrame& frame_1);
private:

	//virtual bool step();
//	virtual std::vector<core::pFrame> do_single_step(const std::vector<core::pFrame>&);
	virtual std::vector<core::pFrame> do_special_step(const param_type&) override;
	virtual bool set_param(const core::Parameter& param);
	bool do_process_event(const std::string& event_name, const event::pBasicEvent& event);
//	core::pBasicFrame frame_0;
//	core::pBasicFrame frame_1;
	ssize_t x_;
	ssize_t y_;
};

} /* namespace overlay */
} /* namespace yuri */
#endif /* OVERLAY_H_ */
