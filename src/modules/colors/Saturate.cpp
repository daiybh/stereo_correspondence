/*!
 * @file 		Saturate.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		06.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Saturate.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/utils/assign_events.h"
namespace yuri {
namespace saturate {


IOTHREAD_GENERATOR(Saturate)

MODULE_REGISTRATION_BEGIN("saturate")
		REGISTER_IOTHREAD("saturate",Saturate)
MODULE_REGISTRATION_END()

core::Parameters Saturate::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Saturate");
	p["saturate"]["Saturation of the image. 1.0 is original, 0.0 will be black and white."]=1.0;
	p["crop"]["Crop the resulting values to correct ranges. Setting to false will allow values to over/underflow"]=true;
	return p;
}


Saturate::Saturate(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("saturate")),
BasicEventConsumer(log),
saturation_(1.0),crop_(true)
{
	IOTHREAD_INIT(parameters)

	using namespace core::raw_format;
	set_supported_formats({yuyv422, yvyu422, uyvy422, vyuy422, yuv444, yuv411, yuva4444, ayuv4444});
}

Saturate::~Saturate() noexcept
{
}


namespace {
template<bool crop>
struct convert_color;

template<>
struct convert_color<false>{
	template<typename T>
	static
	typename std::enable_if<std::is_unsigned<T>::value,T>::type
	eval(T orig, double saturation) {
		const auto mid_value = std::numeric_limits<T>::max() >> 1;
		return static_cast<T>((static_cast<double>(orig) - mid_value) * saturation + mid_value);
	}
};

template<typename T>
T crop(double val) {
	const auto min_value = std::numeric_limits<T>::min();
	const auto max_value = std::numeric_limits<T>::max();
	const auto fmin_value = static_cast<double>(min_value);
	const auto fmax_value = static_cast<double>(max_value);
	if (fmin_value > val) return min_value;
	if (fmax_value < val) return max_value;
	return static_cast<T>(val);
}

template<>
struct convert_color<true>{
	template<typename T>
	static
	typename std::enable_if<std::is_unsigned<T>::value,T>::type
	eval(T orig, double saturation) {
		const auto mid_value = std::numeric_limits<T>::max() >> 1;
		return crop<T>((static_cast<double>(orig) - mid_value) * saturation + mid_value);
	}
};

template<bool>
struct keep_color{
	template<typename T>
	static T eval(T orig, double) {
		return orig;
	}
};

template<typename T, bool>
void process_pixels(T&, T&, double ) {

}

template<typename T, bool crop, template<bool> class C, template<bool> class... Rest>
void process_pixels(T& in, T& out, double saturation) {
	*out++ = C<crop>::eval(*in++, saturation);
	process_pixels<T, crop, Rest...>(in, out, saturation);
}

template<bool crop, template<bool> class... Converters>
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

template<bool crop>
core::pRawVideoFrame convert_frame_dispatch2(const core::pRawVideoFrame& frame, double saturation)
{
	const auto fmt = frame->get_format();
	using namespace core::raw_format;
	switch (fmt) {
		case yuyv422:
		case yvyu422:
			return convert_frame<crop, keep_color, convert_color>(frame, saturation);
		case uyvy422:
		case vyuy422:
			return convert_frame<crop, convert_color, keep_color>(frame, saturation);
		case yuv444:
			return convert_frame<crop, keep_color, convert_color, convert_color>(frame, saturation);
		case yuv411:
		case yvu411:
			return convert_frame<crop, keep_color, keep_color, convert_color>(frame, saturation);
		case ayuv4444:
			return convert_frame<crop, keep_color, keep_color, convert_color, convert_color>(frame, saturation);
		case yuva4444:
			return convert_frame<crop, keep_color, convert_color, convert_color, keep_color>(frame, saturation);
		default:
			return {};
	}
}


core::pRawVideoFrame convert_frame_dispatch(const core::pRawVideoFrame& frame, double saturation, bool crop)
{
	if (crop) return convert_frame_dispatch2<true>(frame, saturation);
	return convert_frame_dispatch2<false>(frame, saturation);
}

}

core::pFrame Saturate::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
	return convert_frame_dispatch(frame, saturation_, crop_);
}

bool Saturate::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(saturation_, "saturation")
			(crop_, "crop"))
		return true;
	return base_type::set_param(param);
}
bool Saturate::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			(saturation_, "saturation")
			(crop_, "crop"))
		return true;
	return false;
}

} /* namespace saturate */
} /* namespace yuri */
