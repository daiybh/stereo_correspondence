/*!
 * @file 		manipulate_colors.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef MANIPULATE_COLORS_H_
#define MANIPULATE_COLORS_H_
#include <limits>
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/irange.h"

namespace yuri {
namespace colors {
namespace {
using namespace core::raw_format;
const std::vector<format_t> color_supported_formats =
		{yuyv422, yvyu422, uyvy422, vyuy422, yuv444, yuv411, yuva4444, ayuv4444};

}
template<bool crop>
struct crop_value {};

template<>
struct crop_value<true>{
template<typename T>
static T eval(double val) {
	const auto min_value = std::numeric_limits<T>::min();
	const auto max_value = std::numeric_limits<T>::max();
	const auto fmin_value = static_cast<double>(min_value);
	const auto fmax_value = static_cast<double>(max_value);
	if (fmin_value > val) return min_value;
	if (fmax_value < val) return max_value;
	return static_cast<T>(val);
}
};

template<>
struct crop_value<false>{
template<typename T>
static T eval(double val) {
	return static_cast<T>(val);
}
};

template<typename T, bool>
void process_pixels(T&, T&, double ) {

}

template<typename T, bool crop, class C, class... Rest>
void process_pixels(T& in, T& out, double saturation) {
	*out++ = crop_value<crop>::template eval<typename std::decay<decltype(*in)>::type>(C::eval(*in++, saturation));
	process_pixels<T, crop, Rest...>(in, out, saturation);
}

template<bool crop, class... Converters>
core::pRawVideoFrame convert_frame(const core::pRawVideoFrame& frame, double saturation)
{
	const auto res = frame->get_resolution();
	const auto fmt = frame->get_format();
	const auto linesize = PLANE_DATA(frame,0).get_line_size();

	auto out_frame = core::RawVideoFrame::create_empty(fmt, res);
	const auto linesize_out = PLANE_DATA(out_frame,0).get_line_size();
	for (auto line: irange(0, res.height)) {
		auto in = PLANE_DATA(frame, 0).begin() + line * linesize;
		auto out = PLANE_DATA(out_frame, 0).begin() + line * linesize_out;
		const auto in_end = in + std::min(linesize_out, linesize);
		while(in < in_end) {
			process_pixels<typename std::decay<decltype(in)>::type, crop,Converters...>(in, out, saturation);
		}
	}
	return out_frame;
}


template<bool crop, class Lum, class Col>
core::pRawVideoFrame convert_frame_dispatch2(const core::pRawVideoFrame& frame, double saturation)
{
	const auto fmt = frame->get_format();
	using namespace core::raw_format;
	switch (fmt) {
		case yuyv422:
		case yvyu422:
			return convert_frame<crop, Lum, Col>(frame, saturation);
		case uyvy422:
		case vyuy422:
			return convert_frame<crop, Col, Lum>(frame, saturation);
		case yuv444:
			return convert_frame<crop, Lum, Col, Col>(frame, saturation);
		case yuv411:
		case yvu411:
			return convert_frame<crop, Lum, Lum, Col>(frame, saturation);
		case ayuv4444:
			return convert_frame<crop, Lum, Lum, Col, Col>(frame, saturation);
		case yuva4444:
			return convert_frame<crop, Lum, Col, Col, Lum>(frame, saturation);
		default:
			return {};
	}
}


template<class Lum, class Col>
core::pRawVideoFrame convert_frame_dispatch(const core::pRawVideoFrame& frame, double saturation, bool crop)
{
	if (crop) return convert_frame_dispatch2<true, Lum, Col>(frame, saturation);
	return convert_frame_dispatch2<false, Lum, Col>(frame, saturation);
}



struct keep_color{
	template<typename T>
	static T eval(T orig, double) {
		return orig;
	}
};

struct multiply_color{
	template<typename T>
	static
	typename std::enable_if<std::is_unsigned<T>::value, double>::type
	eval(T orig, double value) {
		const auto mid_value = std::numeric_limits<T>::max() >> 1;
		return (static_cast<double>(orig) - mid_value) * value + mid_value;
	}
};



}
}


#endif /* MANIPULATE_COLORS_H_ */
