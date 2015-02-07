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
#include "yuri/core/utils/assign_events.h"
#include "manipulate_colors.h"
namespace yuri {
namespace colors {


IOTHREAD_GENERATOR(Saturate)


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

	set_supported_formats(color_supported_formats);
}

Saturate::~Saturate() noexcept
{
}


core::pFrame Saturate::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
	return convert_frame_dispatch<keep_color, multiply_color>(frame, saturation_, crop_);
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
