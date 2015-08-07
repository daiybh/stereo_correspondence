/*!
 * @file 		ColorPicker.cpp
 * @author 		<Your name>
 * @date		15.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "ColorPicker.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils/assign_events.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/color.h"
#include <numeric>
#include <array>
namespace yuri {
namespace color_picker {


IOTHREAD_GENERATOR(ColorPicker)

MODULE_REGISTRATION_BEGIN("color_picker")
		REGISTER_IOTHREAD("color_picker",ColorPicker)
MODULE_REGISTRATION_END()

core::Parameters ColorPicker::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("ColorPicker");
	p["geometry"]["Rectangle to get the color from"]=geometry_t{10,10,0,0};
	return p;
}


ColorPicker::ColorPicker(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("color_picker")),BasicEventConsumer(log),
event::BasicEventProducer(log),
geometry_(geometry_t{10,10,0,0}),show_color_(true)
{
	IOTHREAD_INIT(parameters)
	using namespace core::raw_format;
	set_supported_formats({rgb24, bgr24, rgb48, bgr48,
							rgba32, bgra32, argb32, abgr32,
							rgba64, bgra64, argb64, abgr64,
							yuyv422, yvyu422, uyvy422, vyuy422,
							yuv444, ayuv4444, yuva4444,

							r8,
							r16,
							g8,
							g16,
							b8,
							b16,
							y8,
							y16,
							u8,
							u16,
							v8,
							v16,
//							depth8,
//							depth16,

	});
}

ColorPicker::~ColorPicker() noexcept
{
}

namespace {
template<typename T, size_t size>
std::array<T, size> average_simple(const core::pRawVideoFrame& frame, const geometry_t& rect, size_t line_size)
{
	const size_t pixels = rect.width * rect.height;
	using pixel_type = std::array<T, size>;
	using acc_type = std::array<size_t, size>;
	pixel_type pixel_out;
	if (pixels) {
		acc_type vals;
		std::fill(vals.begin(), vals.end(), 0);
		auto it = reinterpret_cast<pixel_type *>(PLANE_RAW_DATA(frame,0))+rect.y*line_size + rect.x;
		for (dimension_t line = 0; line < rect.height; ++line) {
			vals = std::accumulate(it,it+rect.width,vals,[](const acc_type& a, const pixel_type& b)
					{
						acc_type acc;
						std::transform(a.begin(), a.end(), b.begin(), acc.begin(),[](const size_t&x, const T&y){return x+y;});
						return acc;
					}
			);
			it+=line_size;
		}
		std::transform(vals.begin(), vals.end(), pixel_out.begin(), [pixels](const size_t& x){return static_cast<T>(x/pixels);});
	} else {
		std::fill(pixel_out.begin(), pixel_out.end(), 0);
	}
	return pixel_out;
}

template<typename T, size_t size>
core::pRawVideoFrame draw_color(const core::pRawVideoFrame& frame, const geometry_t& rect, size_t line_size, const std::array<T, size>& avg)
{
	using pixel_type = std::array<T, size>;
	auto out_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(frame->get_copy());
	auto it_out = reinterpret_cast<pixel_type *>(PLANE_RAW_DATA(out_frame,0))+rect.y*line_size+ rect.x;
	for (dimension_t line = 0; line < rect.height; ++line) {
		std::fill(it_out,it_out+rect.width,avg);
		it_out+=line_size;
	}
	return out_frame;
}

template<format_t>
struct get_color {};

#include "color_kernels.impl"


template<typename T, size_t size, format_t fmt>
std::tuple<core::pRawVideoFrame, core::color_t> process_colors(const core::pRawVideoFrame& frame, const geometry_t& rect, bool draw)
{
	const auto line_size = frame->get_resolution().width;
	const auto avg = average_simple<T, size>(frame, rect, line_size);
	core::pRawVideoFrame frame_out;
	if (draw) {
		frame_out = draw_color(frame, rect, line_size, avg);
	} else frame_out = frame;
	return std::make_tuple(std::move(frame_out), get_color<fmt>::eval(avg));
}


template<typename T, format_t fmt>
std::tuple<core::pRawVideoFrame, core::color_t> process_colors_yuv(const core::pRawVideoFrame& frame, geometry_t rect, bool draw)
{
	const auto line_size = frame->get_resolution().width /2;
	rect.width/=2; rect.x/=2;
	auto avg = average_simple<T, 4>(frame, rect, line_size);
	core::pRawVideoFrame frame_out;
	if (draw) {
		frame_out = draw_color<T, 4>(frame, rect, line_size, avg);
	} else {
		frame_out = frame;
	}
	return std::make_tuple(std::move(frame_out), get_color<fmt>::eval(avg));
}


}

core::pFrame ColorPicker::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
	using namespace core::raw_format;
	const format_t format = frame->get_format();
	const resolution_t resolution = frame->get_resolution();
	core::pRawVideoFrame out_frame = frame;
	const auto rect = intersection(resolution, geometry_);
	core::color_t color;


	switch (format) {
		case rgb24:
			std::tie(out_frame, color) = process_colors<uint8_t, 3, rgb24>(frame, rect, show_color_);
			break;
		case bgr24:
			std::tie(out_frame, color) = process_colors<uint8_t, 3, bgr24>(frame, rect, show_color_);
			break;
		case rgb48:
			std::tie(out_frame, color) = process_colors<uint16_t, 3, rgb48>(frame, rect, show_color_);
			break;
		case bgr48:
			std::tie(out_frame, color) = process_colors<uint16_t, 3, bgr48>(frame, rect, show_color_);
			break;

		case rgba32:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, rgba32>(frame, rect, show_color_);
			break;
		case bgra32:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, bgra32>(frame, rect, show_color_);
			break;
		case argb32:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, argb32>(frame, rect, show_color_);
			break;
		case abgr32:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, abgr32>(frame, rect, show_color_);
			break;

		case rgba64:
			std::tie(out_frame, color) = process_colors<uint16_t, 4, rgba64>(frame, rect, show_color_);
			break;
		case bgra64:
			std::tie(out_frame, color) = process_colors<uint16_t, 4, bgra64>(frame, rect, show_color_);
			break;
		case argb64:
			std::tie(out_frame, color) = process_colors<uint16_t, 4, argb64>(frame, rect, show_color_);
			break;
		case abgr64:
			std::tie(out_frame, color) = process_colors<uint16_t, 4, abgr64>(frame, rect, show_color_);
			break;

		case yuyv422:
			std::tie(out_frame, color) = process_colors_yuv<uint8_t, yuyv422>(frame, rect, show_color_);
			break;
		case yvyu422:
			std::tie(out_frame, color) = process_colors_yuv<uint8_t, yvyu422>(frame, rect, show_color_);
			break;
		case uyvy422:
			std::tie(out_frame, color) = process_colors_yuv<uint8_t, uyvy422>(frame, rect, show_color_);
			break;
		case vyuy422:
			std::tie(out_frame, color) = process_colors_yuv<uint8_t, vyuy422>(frame, rect, show_color_);
			break;
		case yuv444:
			std::tie(out_frame, color) = process_colors<uint8_t, 3, yuv444>(frame, rect, show_color_);
			break;
		case ayuv4444:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, ayuv4444>(frame, rect, show_color_);
			break;
		case yuva4444:
			std::tie(out_frame, color) = process_colors<uint8_t, 4, yuva4444>(frame, rect, show_color_);
			break;
		// Single component
		case r8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, r8>(frame, rect, show_color_);
			break;
		case r16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, r16>(frame, rect, show_color_);
			break;
		case g8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, g8>(frame, rect, show_color_);
			break;
		case g16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, g16>(frame, rect, show_color_);
			break;
		case b8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, b8>(frame, rect, show_color_);
			break;
		case b16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, b16>(frame, rect, show_color_);
			break;
		case y8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, y8>(frame, rect, show_color_);
			break;
		case y16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, y16>(frame, rect, show_color_);
			break;
		case u8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, u8>(frame, rect, show_color_);
			break;
		case u16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, u16>(frame, rect, show_color_);
			break;
		case v8:
			std::tie(out_frame, color) = process_colors<uint8_t, 1, v8>(frame, rect, show_color_);
			break;
		case v16:
			std::tie(out_frame, color) = process_colors<uint16_t, 1, v16>(frame, rect, show_color_);
			break;
//		case depth8:
//			std::tie(out_frame, color) = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("depth",e);this->emit_event("d",e);});
//			break;
//		case depth16:
//			std::tie(out_frame, color) = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("depth",e);this->emit_event("d",e);});
//			break;
		default:
			return out_frame;

	}
	log[log::verbose_debug] << "Found color: " << color;
	emit_event("color", lexical_cast<std::string>(color));
	return out_frame;
}
bool ColorPicker::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(geometry_, "geometry"))
		return true;
	return base_type::set_param(param);
}


bool ColorPicker::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			.vector_values("geometry", geometry_.x, geometry_.y, geometry_.width, geometry_.height)
			.vector_values("geometry", geometry_.x, geometry_.y)
			(geometry_, 		"geometry")
			(geometry_.x, 		"x")
			(geometry_.y, 		"y")
			(geometry_.width, 	"width")
			(geometry_.height, 	"height"))
		return true;
	return false;
}

} /* namespace color_picker */
} /* namespace yuri */
