/*!
 * @file 		Scale.h
 * @author 		<Your name>
 * @date 		09.04.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef SCALE_H_
#define SCALE_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace scale {

class Scale: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Scale(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Scale() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;

	resolution_t resolution_;
};

} /* namespace scale */
} /* namespace yuri */
#endif /* SCALE_H_ */
