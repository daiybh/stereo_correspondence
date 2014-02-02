/*!
 * @file 		UVRTDxt.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVRTDXT_H_
#define UVRTDXT_H_

#include "yuri/ultragrid/UVVideoCompress.h"
namespace yuri {
namespace uv_rtdxt {



class UVRTDxt: public ultragrid::UVVideoCompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVRTDxt(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVRTDxt() noexcept;
private:
	
//	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	format_t format_;
//	module* encoder_;
};

} /* namespace uv_rtdxt */
} /* namespace yuri */
#endif /* UVRTDXT_H_ */
