/*!
 * @file 		Pad.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		14.08.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PAD_H_
#define PAD_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

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
class Pad: public core::SpecializedIOFilter<core::RawVideoFrame>
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Pad(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Pad() noexcept;
private:

	virtual core::pFrame		do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param);

	resolution_t				resolution_;
//	size_t 						width_;
//	size_t 						height_;
	horizontal_alignment_t		halign_;
	vertical_alignment_t		valign_;
};

} /* namespace pad */
} /* namespace yuri */
#endif /* PAD_H_ */
