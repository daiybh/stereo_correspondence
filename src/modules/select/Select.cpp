/*!
 * @file 		Select.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		15.03.2014
 * @copyright	Institute of Intermedia, 2014
 * 				Distributed BSD License
 *
 */

#include "Select.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"
#include "yuri/core/utils/assign_events.h"

namespace yuri {
namespace select {


IOTHREAD_GENERATOR(Select)


core::Parameters Select::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Select has single output and multiple inputs. Parameter or event 'index' selects, which input will be passed through.");
	p["index"]["input to pass through"]=0;
	return p;
}


Select::Select(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,1,std::string("select")),BasicEventConsumer(log),index_(0)
{
	IOTHREAD_INIT(parameters)
}

Select::~Select() noexcept
{
}

bool Select::step()
{
	process_events();
	push_frame(0, pop_frame(index_));
	return true;
}
bool Select::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(index_, "index"))
		return true;
	return base_type::set_param(param);
}

void Select::do_connect_in(position_t pos, core::pPipe pipe)
{
	position_t inp = do_get_no_in_ports();
	if (pos < 0) {
		resize(inp+1,1);
		pos = inp;
	} else if (pos >= inp) {
		resize(pos+1,1);
	}
	base_type::do_connect_in(pos, pipe);
}

bool Select::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			(index_, "index"))
		return true;
	return false;
}

} /* namespace select */
} /* namespace yuri */
