/*!
 * @file 		Crop.cpp
 * @author 		Zdenek Travnicek
 * @date 		17.11.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Crop.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"

namespace yuri {

namespace io {

MODULE_REGISTRATION_BEGIN("crop")
		REGISTER_IOTHREAD("crop",Crop)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(Crop)

core::Parameters Crop::configure()
{
	core::Parameters p  = base_type::configure();
	p.set_description("Crops the image to the specified dimensions");
	p["geometry"]["Geometry to crop"]=geometry_t{800,600,0,0};
	return p;
}

namespace {
bool verify_support(const core::raw_format::raw_format_t& fmt)
{
	if (fmt.planes.size() != 1) return false;
	const auto& plane= fmt.planes[0];
	if (plane.components.empty()) return false;
	const auto& depth = plane.bit_depth;
	if (depth.first % (depth.second * 8)) return false;

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
Crop::Crop(log::Log &log_, core::pwThreadBase parent,const core::Parameters &parameters):
	base_type(log_,parent,"Crop"),event::BasicEventConsumer(log),geometry_{800,600,0,0}
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(get_supported_fmts(log));

}

Crop::~Crop() noexcept {

}


core::pFrame Crop::do_special_single_step(const core::pRawVideoFrame& frame)
{
	const format_t format = frame->get_format();
	const auto& fi = core::raw_format::get_format_info(format);
	if (!verify_support(fi)) return {};


	const resolution_t in_res = frame->get_resolution();
	const geometry_t geometry_out = intersection(in_res, geometry_);

	log[log::verbose_debug] << "Cropping to " << geometry_out;

	const auto depth = fi.planes[0].bit_depth;
	const size_t bpp = depth.first/depth.second/8;
	const dimension_t copy_bytes = geometry_out.width * bpp;
	const dimension_t line_size = in_res.width * bpp;
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(format, geometry_out.get_resolution());
	auto iter_in = PLANE_DATA(frame,0).begin() +  geometry_out.x * bpp + geometry_out.y * line_size;
	auto iter_out = PLANE_DATA(frame_out,0).begin();
	for (dimension_t line = 0; line < geometry_out.height; ++line) {
		std::copy(iter_in, iter_in+copy_bytes, iter_out);
		iter_in += line_size;
		iter_out += copy_bytes;
	}

	return frame_out;
}

bool Crop::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name()== "geometry") {
		geometry_=parameter.get<geometry_t>();
	} else  return base_type::set_param(parameter);
	return true;
}
bool Crop::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	try {
		if (event_name == "geometry") {
			geometry_ = event::lex_cast_value<geometry_t>(event);
		} else if (event_name == "x") {
			geometry_.x = event::lex_cast_value<position_t>(event);
		} else if (event_name == "y") {
			geometry_.y = event::lex_cast_value<position_t>(event);
		} else if (event_name == "width") {
			geometry_.width = event::lex_cast_value<dimension_t>(event);
		} else if (event_name == "height") {
			geometry_.height = event::lex_cast_value<dimension_t>(event);
		} else return false;
	}
	catch (std::bad_cast&) {
		log[log::info] << "bad cast in " << event_name;
		return false;
	}
	return true;
}
}
}
