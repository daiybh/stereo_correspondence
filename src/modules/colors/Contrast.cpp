/*!
 * @file 		Contrast.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Contrast.h"
#include "yuri/core/Module.h"
#include "manipulate_colors.h"
#include "yuri/core/utils/assign_events.h"
namespace yuri {
namespace colors {


IOTHREAD_GENERATOR(Contrast)

core::Parameters Contrast::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Contrast");
	p["contrast"]["Value for contrast, values greater than zero increase the contrast, lower decrease."]=1.0;
	p["crop"]["Crop the resulting values to correct ranges. Setting to false will allow values to over/underflow"]=true;
	return p;
}


Contrast::Contrast(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("contrast")),
BasicEventConsumer(log),
contrast_(1.0),crop_(true)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(color_supported_formats);
}

Contrast::~Contrast() noexcept
{
}

core::pFrame Contrast::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
	return convert_frame_dispatch<multiply_color, keep_color>(frame, contrast_, crop_);
}


bool Contrast::set_param(const core::Parameter& param)
{
if (assign_parameters(param)
			(contrast_, "contrast")
			(crop_, "crop"))
		return true;
	return base_type::set_param(param);
}
bool Contrast::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			(contrast_, "contrast")
			(crop_, "crop"))
		return true;
	return false;
}

} /* namespace contrast */
} /* namespace yuri */
