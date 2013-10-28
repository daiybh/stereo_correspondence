/*!
 * @file 		ColorKey.h
 * @author 		<Your name>
 * @date 		27.05.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef COLORKEY_H_
#define COLORKEY_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"


namespace yuri {
namespace color_key {

enum diff_types_ {
	linear,
	quadratic
};


class ColorKey: public core::SpecializedIOFilter<core::RawVideoFrame>
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ColorKey(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ColorKey() noexcept;
private:

//	virtual bool step();
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	template<class kernel>
	core::pRawVideoFrame find_key(const core::pRawVideoFrame& frame);
	template<format_t format>
	core::pRawVideoFrame dispatch_find_key(const core::pRawVideoFrame& frame);
	uint8_t r_, g_, b_;
	ssize_t delta_, delta2_;
	diff_types_ diff_type_;
};

} /* namespace color_key */
} /* namespace yuri */
#endif /* COLORKEY_H_ */
