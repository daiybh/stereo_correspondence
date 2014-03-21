/*!
 * @file 		ConvertPlanes.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		30.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
//	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t i = 0; i < planes; ++i) {
		//iters[i]=iter_out+i;
		iters[i]=PLANE_DATA(frame_out, i).begin();
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

template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_sub3_xy(core::pRawVideoFrame frame) {
//	printf("BOO1\n");
	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
//	std::vector<iter_t> iters(planes);
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		iter_t it1 = iter_in1;
		iter_t it2 = iter_in2;
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			for (size_t col = 0; col < res.width; col+=2) {
				for (size_t col2 = col; col2 < std::min(res.width,col+2); ++col2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1;
					*iter_out++ = *it2;
				}
				it1++;it2++;
			}
		}
		iter_in1++;iter_in2++;
	}
	return frame_out;
}
//template<format_t in, format_t out>
//core::pRawVideoFrame merge_planes_411p_422(core::pRawVideoFrame frame) {
//	const resolution_t res = frame->get_resolution();
//	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
//	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
//	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
//	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
//	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
//	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
//	for (size_t line = 0; line < res.height; ++line) {
//		for (size_t col = 0; col < res.width; col+=4) {
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in1;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in2;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in1++;
//				*iter_out++ = *iter_in0++;
//				*iter_out++ = *iter_in2++;
//		}
//	}
//	return frame_out;
//}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_yuyv(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1++;
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it2++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_yvyu(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it2++;
					*iter_out++ = *iter_in0++;
					*iter_out++ = *it1++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_uyvy(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
				*iter_out++ = *it1++;
				*iter_out++ = *iter_in0++;
				*iter_out++ = *it2++;
				*iter_out++ = *iter_in0++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}
template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_420p_vyuy(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; line+=2) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			iter_t it1 = iter_in1;
			iter_t it2 = iter_in2;
			for (size_t col = 0; col < res.width; col+=2) {
				*iter_out++ = *it2++;
				*iter_out++ = *iter_in0++;
				*iter_out++ = *it1++;
				*iter_out++ = *iter_in0++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}

	return frame_out;
}

template<format_t in, format_t out>
core::pRawVideoFrame merge_planes_422p_yuyv(core::pRawVideoFrame frame) {

	const resolution_t res = frame->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(out, res);
	typedef decltype(PLANE_DATA(frame_out,0).begin()) iter_t;
	iter_t iter_in0 = PLANE_DATA(frame, 0).begin();
	iter_t iter_in1 = PLANE_DATA(frame, 1).begin();
	iter_t iter_in2 = PLANE_DATA(frame, 2).begin();
	iter_t iter_out = PLANE_DATA(frame_out, 0).begin();
	for (size_t line = 0; line < res.height; ++line) {
		for (size_t line2 = line; line2 < std::min(res.height,line+2); ++line2) {
			for (size_t col = 0; col < res.width; col+=2) {
				*iter_out++ = *iter_in0++;
				*iter_out++ = *iter_in1++;
				*iter_out++ = *iter_in0++;
				*iter_out++ = *iter_in2++;
			}
		}
		iter_in1+=res.width/2;
		iter_in2+=res.width/2;
	}
	return frame_out;
}

core::pFrame dispatch(core::pRawVideoFrame frame, format_t target) {
	if (!frame) return {};
	format_t source = frame->get_format();
	using namespace yuri::core::raw_format;
//	printf("BOO0\n");
//	if (source == rgb24 && target == rgb24p) return split_planes<rgb24, rgb24p, 3>(frame);
//	if (source == rgba32 && target == rgba32p) return split_planes<rgba32, rgba32p, 4>(frame);
	if (source == yuv444 && target == yuv444p) return split_planes<yuv444, yuv444p, 3>(frame);
//	if (source == yuv420p && target == yuv444) return merge_planes_sub3_xy<yuv420p, yuv444>(frame);
//	if (source == yuv411p && target == yuyv422) return merge_planes_411p_422<yuv420p, yuyv422>(frame);
	if (source == yuv420p && target == yuyv422) return merge_planes_420p_yuyv<yuv420p, yuyv422>(frame);
	if (source == yuv420p && target == yvyu422) return merge_planes_420p_yvyu<yuv420p, yvyu422>(frame);
	if (source == yuv420p && target == uyvy422) return merge_planes_420p_uyvy<yuv420p, uyvy422>(frame);
	if (source == yuv420p && target == vyuy422) return merge_planes_420p_vyuy<yuv420p, vyuy422>(frame);
	if (source == yuv422p && target == yuyv422) return merge_planes_422p_yuyv<yuv422p, yuyv422>(frame);
	return {};
}



}
IOTHREAD_GENERATOR(ConvertPlanes)

MODULE_REGISTRATION_BEGIN("convert_planar")
		REGISTER_IOTHREAD("convert_planar",ConvertPlanes)
//		REGISTER_CONVERTER(yuri::core::raw_format::rgb24, yuri::core::raw_format::rgb24p, "convert_planar", 5)
//		REGISTER_CONVERTER(yuri::core::raw_format::rgba32, yuri::core::raw_format::rgba32p, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv444, yuri::core::raw_format::yuv444p, "convert_planar", 5)
//		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yuv444, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yuyv422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::yvyu422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::uyvy422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv420p, yuri::core::raw_format::vyuy422, "convert_planar", 5)
		REGISTER_CONVERTER(yuri::core::raw_format::yuv422p, yuri::core::raw_format::yuyv422, "convert_planar", 5)

MODULE_REGISTRATION_END()

core::Parameters ConvertPlanes::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("ConvertPlanes");
	p["format"]["Target format"]="YUV";
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
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) {
		log[log::warning] << "Got bad frame type!!";
		return {};
	}
	return dispatch(frame, target_format);
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
