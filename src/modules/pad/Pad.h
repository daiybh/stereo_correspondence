/*!
 * @file 		Pad.h
 * @author 		<Your name>
 * @date 		14.08.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef PAD_H_
#define PAD_H_

#include "yuri/core/BasicIOFilter.h"

namespace yuri {
namespace pad {

enum class horizontal_alignment_t{
	left,
	center,
	right
};
enum class vertical_alignment_t{
	top,
	center,
	bottom
};
class Pad: public core::BasicIOFilter
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Pad();
private:
	Pad(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual core::pBasicFrame		do_simple_single_step(const core::pBasicFrame& frame);
	virtual bool set_param(const core::Parameter& param);

	size_t 						width_;
	size_t 						height_;
	horizontal_alignment_t		halign_;
	vertical_alignment_t		valign_;
};

} /* namespace pad */
} /* namespace yuri */
#endif /* PAD_H_ */
