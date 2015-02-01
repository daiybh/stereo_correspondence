/*!
 * @file 		SplitPlanes.h
 * @author 		<Your name>
 * @date 		31.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef SPLITPLANES_H_
#define SPLITPLANES_H_

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace split_planes {

class SplitPlanes: public core::SpecializedMultiIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	SplitPlanes(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SplitPlanes() noexcept;
private:
	virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame> frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	bool keep_format_;
};

} /* namespace split_planes */
} /* namespace yuri */
#endif /* SPLITPLANES_H_ */
