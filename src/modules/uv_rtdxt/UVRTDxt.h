/*!
 * @file 		UVRTDxt.h
 * @author 		<Your name>
 * @date 		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVRTDXT_H_
#define UVRTDXT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
struct module;
namespace yuri {
namespace uv_rtdxt {



class UVRTDxt: public core::SpecializedIOFilter<core::RawVideoFrame>
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVRTDxt(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVRTDxt() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);

	module* encoder_;
};

} /* namespace uv_rtdxt */
} /* namespace yuri */
#endif /* UVRTDXT_H_ */
