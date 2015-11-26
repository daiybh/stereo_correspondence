/*!
 * @file 		ColorPicker.h
 * @author 		<Your name>
 * @date 		15.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef COLORPICKER_H_
#define COLORPICKER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"

namespace yuri {
namespace color_picker {

class ColorPicker: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventConsumer, public event::BasicEventProducer
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ColorPicker(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ColorPicker() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	geometry_t geometry_;
	bool show_color_;
	resolution_t matrix_;
};

} /* namespace color_picker */
} /* namespace yuri */
#endif /* COLORPICKER_H_ */
