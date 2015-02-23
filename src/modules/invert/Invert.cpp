/*!
 * @file 		Invert.cpp
 * @author 		<Your name>
 * @date		21.02.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Invert.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include <limits>
namespace yuri {
namespace invert {


IOTHREAD_GENERATOR(Invert)

MODULE_REGISTRATION_BEGIN("invert")
		REGISTER_IOTHREAD("invert",Invert)
MODULE_REGISTRATION_END()

core::Parameters Invert::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Invert");
	return p;
}

namespace {
bool verify_support(const core::raw_format::raw_format_t& fmt)
{
	// multiple planes should be implemented as well..
	if (fmt.planes.size() != 1) return false;
	const auto& plane= fmt.planes[0];
	if (plane.components.empty()) return false;
	const auto& depth = plane.bit_depth;
	if (depth.first % (depth.second * 8)) return false;
	const size_t b = plane.component_bit_depths[0];
	if (b != 8 && b!= 16) return false;
	for (size_t b2: plane.component_bit_depths) {
		if (b != b2) return false;
	}

	return true;
}
std::vector<format_t> get_supported_fmts(log::Log& log) {
	std::vector<format_t> fmts;
	for (const auto& f: core::raw_format::formats()) {
		if (verify_support(f.second)) {
			fmts.push_back(f.first);
			log[log::verbose_debug] << "Setting format " << f.second.name << " as supported";
		}
	}
	return fmts;
}
}


Invert::Invert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("invert"))
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(get_supported_fmts(log));
}

Invert::~Invert() noexcept
{
}

namespace {
template<typename T>
void invert_line(const T* start, const T*end, T* out_end)
{
	/* constexpr */ T t_max = std::numeric_limits<T>::max();
	std::transform(start,end,out_end,[t_max](const T& v){return v^t_max;});
}

template<typename T, class T2>
void process_lines(const T2* start, T2* out_start, size_t lines, size_t line_size)
{
	const T* s= reinterpret_cast<const T*>(start);
	T* d= reinterpret_cast<T*>(out_start);
	const size_t size = line_size *sizeof(T2) / sizeof(T);
	for (size_t line = 0; line < lines; ++line) {
		invert_line(s, s + size, d);
		s += size;
		d += size;
	}
}
}

core::pFrame Invert::do_special_single_step(core::pRawVideoFrame frame)
{
	const auto& fi = core::raw_format::get_format_info(frame->get_format());
	if (!verify_support(fi)) return {};

	const auto depth = fi.planes[0].bit_depth;
	const size_t bpp = depth.first/depth.second/8;
	const size_t comp_bpp =  fi.planes[0].component_bit_depths[0];

	const size_t line_size = bpp * frame->get_width();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(frame->get_format(), frame->get_resolution());

	const auto start_in = PLANE_DATA(frame, 0).begin();
	const auto start_out = PLANE_DATA(frame_out, 0).begin();
	switch (comp_bpp) {
		case 8:
			process_lines<uint8_t>(start_in, start_out, frame->get_height(), line_size);
			break;
		case 16:
			process_lines<uint16_t>(start_in, start_out, frame->get_height(), line_size);
			break;
		default:
			return {};
	}

	return frame_out;
}

bool Invert::set_param(const core::Parameter& param)
{
	return base_type::set_param(param);
}

} /* namespace invert */
} /* namespace yuri */
