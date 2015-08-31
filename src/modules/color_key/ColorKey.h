/*!
 * @file 		ColorKey.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		27.05.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef COLORKEY_H_
#define COLORKEY_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/utils/color.h"
namespace yuri {
namespace color_key {

enum diff_types_ {
	linear,
	quadratic
};


class ColorKey: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventConsumer
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ColorKey(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ColorKey() noexcept;
private:

//	virtual bool step();
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	template<class kernel>
	core::pRawVideoFrame find_key(const core::pRawVideoFrame& frame);
	template<format_t format>
	core::pRawVideoFrame dispatch_find_key(const core::pRawVideoFrame& frame);
	core::color_t color_;
	size_t y_cutoff_;
	ssize_t delta_, delta2_;
	diff_types_ diff_type_;
};

} /* namespace color_key */
} /* namespace yuri */
#endif /* COLORKEY_H_ */
