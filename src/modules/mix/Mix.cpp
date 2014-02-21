/*!
 * @file 		Mix.cpp
 * @author 		<Your name>
 * @date		21.02.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Mix.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace mix {


IOTHREAD_GENERATOR(Mix)

MODULE_REGISTRATION_BEGIN("mix")
		REGISTER_IOTHREAD("mix",Mix)
MODULE_REGISTRATION_END()

core::Parameters Mix::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Mix");
	return p;
}


Mix::Mix(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,1,std::string("mix"))
{
	IOTHREAD_INIT(parameters)
	set_latency(1_ms);
}

Mix::~Mix() noexcept
{
}

bool Mix::step()
{
	bool empty = false;
	while (!empty) {
		empty = true;
		for (int i=0;i<get_no_in_ports();++i) {
			if (core::pFrame frame = pop_frame(i)) {
				push_frame(0, frame);
				empty = false;
			}
		}
	}
	return true;
}
void Mix::do_connect_in(position_t pos, core::pPipe pipe)
{
	position_t inp = do_get_no_in_ports();
	if (pos < 0) {
		resize(inp+1,1);
		pos = inp;
	} else if (pos >= inp) {
		resize(pos+1,1);
	}
	IOThread::do_connect_in(pos, pipe);
}
bool Mix::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace mix */
} /* namespace yuri */
