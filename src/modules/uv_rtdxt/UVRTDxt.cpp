/*!
 * @file 		UVRTDxt.cpp
 * @author 		<Your name>
 * @date		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVRTDxt.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
extern "C" {
#include "video_compress.h"
#include "video_compress/dxt_glsl.h"
#include "video_frame.h"
}
namespace yuri {
namespace uv_rtdxt {


IOTHREAD_GENERATOR(UVRTDxt)

MODULE_REGISTRATION_BEGIN("uv_rtdxt")
		REGISTER_IOTHREAD("uv_rtdxt",UVRTDxt)
MODULE_REGISTRATION_END()

core::Parameters UVRTDxt::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVRTDxt");
	return p;
}


UVRTDxt::UVRTDxt(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent, std::string("uv_rtdxt"))
{
	IOTHREAD_INIT(parameters)
	video_compress_params params{"DXT1"};
	if (!(encoder_ = dxt_glsl_compress_init(nullptr,&params))) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create DXT encoder");
	}
}

UVRTDxt::~UVRTDxt() noexcept
{
}

core::pFrame UVRTDxt::do_special_single_step(const core::pRawVideoFrame& frame)
{
	video_frame * uv_frame = ultragrid::allocate_uv_frame(frame);
	video_frame * out_uv_frame  = dxt_glsl_compress(encoder_, uv_frame, 0);
	vf_free(uv_frame);


	if (out_uv_frame) {
		core::pFrame out_frame = ultragrid::copy_from_from_uv(out_uv_frame, log);
		if (out_frame) return {out_frame};
	}
	return {};
}
bool UVRTDxt::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_rtdxt */
} /* namespace yuri */
