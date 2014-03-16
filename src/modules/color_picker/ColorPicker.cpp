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
#include "yuri/event/EventHelpers.h"
#include "yuri/core/frame/raw_frame_types.h"

#include <numeric>
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
geometry_{10,10,0,0},show_color_(true)
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
							depth8,
							depth16,

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
//	const resolution_t resolution = frame->get_resolution();
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

template<typename T, size_t size>
event::pBasicEvent prepare_event(const std::array<T, size>& pixel)
{
	std::vector<event::pBasicEvent> vec(size);
	std::transform(pixel.begin(), pixel.end(), vec.begin(),[](const T& val){return std::make_shared<event::EventInt>(val);});
	return std::make_shared<event::EventVector>(std::move(vec));
}

template<typename T, size_t size, class F>
core::pRawVideoFrame process_colors(const core::pRawVideoFrame& frame, const geometry_t& rect, bool draw, F func)
{
	const auto line_size = frame->get_resolution().width;
	const auto avg = average_simple<T, size>(frame, rect, line_size);
	func(prepare_event(avg));
	if (draw) {
		return draw_color(frame, rect, line_size, avg);
	}
	return frame;
}


template<typename T, size_t ypos, class F>
core::pRawVideoFrame process_colors_yuv(const core::pRawVideoFrame& frame, geometry_t rect, bool draw, F func)
{
	const auto line_size = frame->get_resolution().width /2;
	rect.width/=2; rect.x/=2;
	auto avg = average_simple<T, 4>(frame, rect, line_size);
	const T y = avg[ypos]/2+avg[ypos+2]/2;
	avg[ypos]=y;avg[ypos+2]=y;
	const size_t uvpos = (ypos+1)%2;
	func(prepare_event<T,3>({{y,avg[uvpos],avg[uvpos+2]}}));
	if (draw) {
		return draw_color<T, 4>(frame, rect, line_size, avg);
	}
	return frame;
}


}

core::pFrame ColorPicker::do_special_single_step(const core::pRawVideoFrame& frame)
{
	process_events();
	using namespace core::raw_format;
	const format_t format = frame->get_format();
	const resolution_t resolution = frame->get_resolution();
	core::pRawVideoFrame out_frame = frame;
	const auto rect = intersection(resolution, geometry_);

	switch (format) {
		case rgb24:
		case bgr24:
			out_frame = process_colors<uint8_t, 3>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("rgb",e);});
			break;
		case rgb48:
		case bgr48:
			out_frame = process_colors<uint16_t, 3>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("rgb",e);});
			break;

		case rgba32:
		case bgra32:
		case argb32:
		case abgr32:
			out_frame = process_colors<uint8_t, 4>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("rgba",e);this->emit_event("rgb",e);});
			break;

		case rgba64:
		case bgra64:
		case argb64:
		case abgr64:
			out_frame = process_colors<uint16_t, 4>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("rgba",e);this->emit_event("rgb",e);});
			break;

		case yuyv422:
		case yvyu422:
			out_frame = process_colors_yuv<uint8_t,0>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("yuv",e);});
			break;
		case uyvy422:
		case vyuy422:
			out_frame = process_colors_yuv<uint8_t,1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("yuv",e);});
			break;
		case yuv444:
			out_frame = process_colors<uint8_t, 3>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("yuv",e);});
			break;
		case ayuv4444:
		case yuva4444:
			out_frame = process_colors<uint8_t, 4>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("yuva",e);this->emit_event("yuv",e);});
			break;
		// Single component
		case r8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("red",e);this->emit_event("r",e);});
			break;
		case r16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("red",e);this->emit_event("r",e);});
			break;
		case g8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("green",e);this->emit_event("g",e);});
			break;
		case g16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("green",e);this->emit_event("g",e);});
			break;
		case b8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("blue",e);this->emit_event("b",e);});
			break;
		case b16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("blue",e);this->emit_event("b",e);});
			break;
		case y8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("y",e);});
			break;
		case y16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("y",e);});
			break;
		case u8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("u",e);});
			break;
		case u16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("u",e);});
			break;
		case v8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("v",e);});
			break;
		case v16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("v",e);});
			break;
		case depth8:
			out_frame = process_colors<uint8_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("depth",e);this->emit_event("d",e);});
			break;
		case depth16:
			out_frame = process_colors<uint16_t, 1>(frame, rect, show_color_, [&](const event::pBasicEvent& e)mutable{this->emit_event("depth",e);this->emit_event("d",e);});
			break;

	}
	return out_frame;
}
bool ColorPicker::set_param(const core::Parameter& param)
{
	if (param.get_name() == "geometry") {
		geometry_ = param.get<geometry_t>();
	} else return base_type::set_param(param);
	return true;
}


bool ColorPicker::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	try {
		if (event_name == "geometry") {
			if (event->get_type() == event::event_type_t::vector_event) {
				auto vec = event::get_value<event::EventVector>(event);
				if (vec.size()<2) {
					throw std::runtime_error("Not enough values to process");
				}
				if (vec.size()<4) {
					geometry_.x = event::lex_cast_value<position_t>(vec[0]);
					geometry_.y = event::lex_cast_value<position_t>(vec[1]);
				} else {
					geometry_.x = event::lex_cast_value<position_t>(vec[0]);
					geometry_.y = event::lex_cast_value<position_t>(vec[1]);
					geometry_.width = event::lex_cast_value<dimension_t>(vec[2]);
					geometry_.height = event::lex_cast_value<dimension_t>(vec[3]);
				}
			} else {
				geometry_ = event::lex_cast_value<geometry_t>(event);
			}
		} else if (event_name == "x") {
			geometry_.x = event::lex_cast_value<position_t>(event);
		} else if (event_name == "y") {
			geometry_.y = event::lex_cast_value<position_t>(event);
		} else if (event_name == "width") {
			geometry_.width = event::lex_cast_value<dimension_t>(event);
		} else if (event_name == "height") {
			geometry_.height= event::lex_cast_value<dimension_t>(event);
		}
	}
	catch (std::exception&) {
		return false;
	}
	return true;
}

} /* namespace color_picker */
} /* namespace yuri */
