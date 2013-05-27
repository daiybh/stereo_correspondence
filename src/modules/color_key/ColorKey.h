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

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace color_key {

enum diff_types_ {
	linear,
	quadratic
};


class ColorKey: public core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~ColorKey();
private:
	ColorKey(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	template<class kernel>
	core::pBasicFrame find_key(const core::pBasicFrame& frame);
	template<format_t format>
	core::pBasicFrame dispatch_find_key(const core::pBasicFrame& frame);
	ubyte_t r_, g_, b_;
	ushort_t delta_, delta2_;
	diff_types_ diff_type_;
};

} /* namespace color_key */
} /* namespace yuri */
#endif /* COLORKEY_H_ */
