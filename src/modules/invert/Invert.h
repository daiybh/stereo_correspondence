/*!
 * @file 		Invert.h
 * @author 		<Your name>
 * @date 		21.02.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef INVERT_H_
#define INVERT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace invert {

class Invert: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Invert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Invert() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
};

} /* namespace invert */
} /* namespace yuri */
#endif /* INVERT_H_ */
