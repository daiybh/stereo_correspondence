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
#include "yuri/core/frame/raw_frame_traits.h"

#include "yuri/core/utils/irange.h"

namespace yuri {
namespace colors {
namespace {
using namespace core::raw_format;
const std::vector<format_t> color_supported_formats =
		{yuyv422, yvyu422, uyvy422, vyuy422, yuv444, yuv411, yuva4444, ayuv4444,
		y8, u8, v8, y16, u16, v16};

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

template<format_t fmt, bool crop, class... Converters>
core::pRawVideoFrame convert_frame(const core::pRawVideoFrame& frame, double saturation)
{
	using data_type = typename core::raw_format::frame_traits<fmt>::component_type;
	using data_pointer = typename std::add_pointer<data_type>::type;

	const auto res = frame->get_resolution();
	const auto linesize = PLANE_DATA(frame,0).get_line_size();
	auto out_frame = core::RawVideoFrame::create_empty(fmt, res);
	const auto linesize_out = PLANE_DATA(out_frame,0).get_line_size();
	for (auto line: irange(0, res.height)) {
		auto in_raw = PLANE_RAW_DATA(frame, 0) + line * linesize;
		data_pointer in = reinterpret_cast<data_pointer>(in_raw);
		data_pointer out = reinterpret_cast<data_pointer>(PLANE_RAW_DATA(out_frame, 0) + line * linesize_out);
		const auto in_end = reinterpret_cast<data_pointer>(in_raw + std::min(linesize_out, linesize));
		while(in < in_end) {
			process_pixels<data_pointer, crop,Converters...>(in, out, saturation);
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
			return convert_frame<yuyv422, crop, Lum, Col>(frame, saturation);
		case yvyu422:
			return convert_frame<yvyu422, crop, Lum, Col>(frame, saturation);
		case uyvy422:
			return convert_frame<uyvy422, crop, Col, Lum>(frame, saturation);
		case vyuy422:
			return convert_frame<vyuy422, crop, Col, Lum>(frame, saturation);
		case yuv444:
			return convert_frame<yuv444, crop, Lum, Col, Col>(frame, saturation);
		case yuv411:
			return convert_frame<yuv411, crop, Lum, Lum, Col>(frame, saturation);
		case yvu411:
			return convert_frame<yvu411, crop, Lum, Lum, Col>(frame, saturation);
		case ayuv4444:
			return convert_frame<ayuv4444, crop, Lum, Lum, Col, Col>(frame, saturation);
		case yuva4444:
			return convert_frame<yuva4444, crop, Lum, Col, Col, Lum>(frame, saturation);
		case y8:
			return convert_frame<y8, crop, Lum>(frame, saturation);
		case u8:
			return convert_frame<u8, crop, Col>(frame, saturation);
		case v8:
			return convert_frame<v8, crop, Col>(frame, saturation);
		case y16:
			return convert_frame<y16, crop, Lum>(frame, saturation);
		case u16:
			return convert_frame<u16, crop, Col>(frame, saturation);
		case v16:
			return convert_frame<v16, crop, Col>(frame, saturation);
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
