/*!
 * @file 		Magnify.cpp
 * @author 		<Your name>
 * @date		31.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Magnify.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/event/EventHelpers.h"

namespace yuri {
namespace magnify {


IOTHREAD_GENERATOR(Magnify)

MODULE_REGISTRATION_BEGIN("magnify")
		REGISTER_IOTHREAD("magnify",Magnify)
MODULE_REGISTRATION_END()

core::Parameters Magnify::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Magnify");
	p["geometry"]["Rectangle to magnify"]=geometry_t{50,50,0,0};
	p["zoom"]["Magnification"]=5;
	return p;
}


Magnify::Magnify(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("magnify")),
event::BasicEventConsumer(log),
geometry_(geometry_t{50,50,0,0}),zoom_(5)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::raw_format::rgb24});
}

Magnify::~Magnify() noexcept
{
}

core::pFrame Magnify::do_special_single_step(const core::pRawVideoFrame& frame)
{
	process_events();
	const resolution_t out_res = {geometry_.width*zoom_, geometry_.height*zoom_};
	auto outframe = core::RawVideoFrame::create_empty(core::raw_format::rgb24, out_res);
	const size_t linesize = PLANE_DATA(frame,0).get_line_size();
	auto out = PLANE_RAW_DATA(outframe, 0);
	for (dimension_t line = 0; line < geometry_.height; ++line) {

		for (size_t z=0;z<zoom_;++z) {
			auto in = PLANE_RAW_DATA(frame, 0) + line * linesize;
			for (dimension_t col = 0; col < geometry_.width; ++col) {
				for (size_t z2=0;z2<zoom_;++z2) {
					std::copy(in, in+3, out);
					out+=3;
				}
				in+=3;
			}
		}
	}


	return outframe;
}
bool Magnify::set_param(const core::Parameter& param)
{
	if (param.get_name() == "geometry") {
		geometry_=param.get<geometry_t>();
	} else if (param.get_name() == "zoom") {
		zoom_ = param.get<size_t>();
	} else return base_type::set_param(param);
	return true;
}

bool Magnify::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
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
	} else if (event_name == "zoom") {
		zoom_ = event::lex_cast_value<size_t>(event);
	}
	return true;
}
} /* namespace magnify */
} /* namespace yuri */
