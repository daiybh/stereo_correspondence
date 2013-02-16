/*!
 * @file 		Dup.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Dup.h"

namespace yuri {

namespace io {

REGISTER("dup",Dup)

IO_THREAD_GENERATOR(Dup)

shared_ptr<Parameters> Dup::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	(*p)["hard_dup"]["Make hard copies of the duplicated frames"]=false;
	p->set_max_pipes(1,-1);
	return p;
}

Dup::Dup(Log &log_, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR
		:BasicIOThread(log_,parent,1,0,"DUP"),hard_dup(false)
{
	IO_THREAD_INIT("Dup")
}

Dup::~Dup() {
}


void Dup::connect_out(int index, shared_ptr<BasicPipe> p)
{
	if (out_ports<index) {
		log[warning]
		    << "Trying to connect pipe to non-existent port. "
		    << "Try to use index -1" << std::endl;
		index=out_ports;
	}
	if (index < 0 ) index = out_ports;
	if (index == out_ports) {
		log[debug] << "Resizing out_portsuts" << std::endl;
		BasicIOThread::resize(1,out_ports+1);
	}
	BasicIOThread::connect_out(index,p);
}

bool Dup::step()
{
	pBasicFrame f;
	if (!in[0]) return true;
	while ((f = in[0]->pop_frame()).get()) {
		log[verbose_debug] << "Read frame with format: " << BasicPipe::get_format_string(f->get_format()) << std::endl;
		const int ports = out_ports;
		if (ports) for (int i = 0; i < ports; ++i) {
			pBasicFrame f2;
			if (i != ports - 1 && hard_dup) f2 = f->get_copy();
			else f2 = f;
			push_raw_frame(i,f2);
		}
	}

	return true;
}

bool Dup::set_param(Parameter &parameter)
{
	if (parameter.name == "hard_dup") {
		hard_dup=parameter.get<bool>();
	} else
		return BasicIOThread::set_param(parameter);
	return true;
}

}
}
