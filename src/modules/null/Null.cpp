/*!
 * @file 		Null.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Null.h"
#include "yuri/core/Module.h"
namespace yuri
{

namespace null
{

REGISTER("null",Null)
IO_THREAD_GENERATOR(Null)

core::pParameters Null::configure()
{
	core::pParameters p = BasicIOThread::configure();
	return p;
}

Null::Null(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters) IO_THREAD_CONSTRUCTOR
		:core::BasicIOThread(_log,parent,1,0,"Null")
{
	IO_THREAD_INIT("Null")
	latency=100000;
}

Null::~Null()
{
}

bool Null::step()
{
	int i=0;
	if (in[0]) while (in[0]->pop_frame()) ++i;
	if (i) log[log::verbose_debug] << "Deleted " << i << " frames" << std::endl;
	return true;
}


}

}
