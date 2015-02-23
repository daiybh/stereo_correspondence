/*!
 * @file 		BayerVisualize.h
 * @author 		<Your name>
 * @date 		31.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef BAYERVISUALIZE_H_
#define BAYERVISUALIZE_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace bayer {

class BayerVisualize: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	BayerVisualize(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~BayerVisualize() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
};

} /* namespace bayer */
} /* namespace yuri */
#endif /* BAYERVISUALIZE_H_ */
