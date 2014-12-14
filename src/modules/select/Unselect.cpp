/*!
 * @file 		Unselect.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		14.12.2014
 * @copyright	Institute of Intermedia, 2014
 * 				Distributed BSD License
 *
 */

#include "Unselect.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"

namespace yuri {
namespace select {


IOTHREAD_GENERATOR(Unselect)

core::Parameters Unselect::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Unselect takes single input and sends to one of outputs (based on parameter or event 'index')");
	p["index"]["output to pass to"]=0;
	return p;
}


Unselect::Unselect(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,0,std::string("Unselect")),BasicEventConsumer(log),index_(0)
{
	IOTHREAD_INIT(parameters)
}

Unselect::~Unselect() noexcept
{
}

bool Unselect::step()
{
	process_events();
	push_frame(index_, pop_frame(0));
	return true;
}
bool Unselect::set_param(const core::Parameter& param)
{
	if (param.get_name() == "index") {
		 index_ = param.get<position_t>();
	} else return base_type::set_param(param);
	return true;
}

void Unselect::do_connect_out(position_t pos, core::pPipe pipe)
{
	log[log::info ] << "Connecting pipe " << pos;
	position_t outp = do_get_no_out_ports();
	if (pos < 0) {
		resize(1,outp+1);
		pos = outp;
	} else if (pos >= outp) {
		resize(1,pos+1);
	}
	base_type::do_connect_out(pos, pipe);
}

bool Unselect::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (event_name == "index") {
		index_ = event::lex_cast_value<position_t>(event);
	}
	return true;
}

} /* namespace select */
} /* namespace yuri */
