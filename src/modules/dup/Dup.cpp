/*!
 * @file 		Dup.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Dup.h"
#include "yuri/core/Module.h"

namespace yuri {

namespace io {

IOTHREAD_GENERATOR(Dup)
MODULE_REGISTRATION_BEGIN("dup")
		REGISTER_IOTHREAD("dup",Dup)
MODULE_REGISTRATION_END()


core::Parameters Dup::configure()
{
	core::Parameters p = MultiIOFilter::configure();
	p["hard_dup"]["Make hard copies of the duplicated frames"]=false;
	//p->set_max_pipes(1,-1);
	return p;
}

Dup::Dup(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
		:MultiIOFilter(log_,parent,1,0,"DUP"),hard_dup_(false)
{
	IOTHREAD_INIT(parameters);
}

Dup::~Dup() noexcept
{
}

void Dup::do_connect_out(position_t index, core::pPipe pipe)
{
	if (index < 0) {
		index = get_no_out_ports();
		resize(1, index+1);
	}
	MultiIOFilter::do_connect_out(index, pipe);
}

std::vector<core::pFrame> Dup::do_single_step(const std::vector<core::pFrame>& frames)
{
	std::vector<core::pFrame> outframes;
	if (!hard_dup_) {
		outframes.resize(get_no_out_ports(), frames[0]);
	} else {
		for (position_t i = 0; i < get_no_out_ports(); ++i) {
			outframes.push_back(frames[0]->get_copy());
		}
	}
	return outframes;
}

bool Dup::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name() == "hard_dup") {
		hard_dup_=parameter.get<bool>();
	} else
		return MultiIOFilter::set_param(parameter);
	return true;
}

}
}
