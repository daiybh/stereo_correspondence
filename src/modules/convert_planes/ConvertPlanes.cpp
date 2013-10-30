/*!
 * @file 		ConvertPlanes.cpp
 * @author 		<Your name>
 * @date		30.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "ConvertPlanes.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/thread/ConverterRegister.h"
namespace yuri {
namespace convert_planar {

namespace {

template<format_t in, format_t out, size_t planes>
core::pRawVideoFrame split_planes(core::pRawVideoFrame frame) {
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	std::vector<iter_t> iters(planes);
	iter_t iter_in = PLANE_DATA(frame, 0).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t i = 0; i < planes; ++i) {
		iters[i]=iter_out+i;
	}
	for (size_t line = 0; line < res.height; ++ line) {
		for (size_t col = 0; col < res.width; ++col) {
			for (size_t i = 0; i < planes; ++i) {
				*iters[i]++=*iter_in++;
			}
		}
	}
	return frame_out;
}


core::pFrame dispatch(core::pRawVideoFrame frame, format_t target) {
	if (!frame) return {};
	format_t source = frame->get_format();
	using namespace yuri::core::raw_format;
	if (source == rgb24 && target == rgb24p) return split_planes<rgb24, rgb24p, 3>(frame);
	if (source == rgba32 && target == rgba32p) return split_planes<rgba32, rgba32p, 4>(frame);
	if (source == yuv444 && target == yuv444p) return split_planes<yuv444, yuv444p, 3>(frame);

	 return {};
}



}
IOTHREAD_GENERATOR(ConvertPlanes)

MODULE_REGISTRATION_BEGIN("convert_planar")
		REGISTER_IOTHREAD("convert_planar",ConvertPlanes)
		REGISTER_CONVERTER(yuri::core::raw_format::rgb24, yuri::core::raw_format::rgb24p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::rgba32, yuri::core::raw_format::rgba32p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv444, yuri::core::raw_format::yuv444p, "convert_planar", 5)

MODULE_REGISTRATION_END()

core::Parameters ConvertPlanes::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("ConvertPlanes");
	p["format"]["Target format"]="YUV444P";
	return p;
}


ConvertPlanes::ConvertPlanes(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("convert_planar"))
{
	IOTHREAD_INIT(parameters)
}

ConvertPlanes::~ConvertPlanes() noexcept
{
}

core::pFrame ConvertPlanes::do_special_single_step(const core::pRawVideoFrame& frame)
{
	return dispatch(frame, format_);
}

core::pFrame ConvertPlanes::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	return dispatch(dynamic_pointer_cast<core::RawVideoFrame>(input_frame), target_format);
}
bool ConvertPlanes::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	}
	return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
}

} /* namespace convert_planar */
} /* namespace yuri */
