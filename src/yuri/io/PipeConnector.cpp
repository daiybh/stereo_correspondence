/*!
 * @file 		PipeConnector.cpp
 * @author 		Zdenek Travnicek
 * @date 		8.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "PipeConnector.h"

namespace yuri {

namespace io {

PipeConnector::PipeConnector(weak_ptr<ThreadBase> thread)
	:thread(thread)
{

}

PipeConnector::PipeConnector(shared_ptr<BasicPipe> pipe, weak_ptr<ThreadBase> thread)
	:thread(thread)
{
	set_pipe(pipe);
}

PipeConnector::PipeConnector(const PipeConnector& orig):
		pipe(orig.pipe),thread(orig.thread)
{
}
PipeConnector::~PipeConnector() {

}

shared_ptr<BasicPipe> PipeConnector::operator ->()
{
	return pipe;
}

void PipeConnector::reset()
{
	pipe.reset();
}

PipeConnector::operator bool()
{
	return pipe;
}

PipeConnector::operator shared_ptr<BasicPipe>()
{
	return pipe;
}

void PipeConnector::reset(shared_ptr<BasicPipe> pipe0)
{
	set_pipe(pipe0);
}

void PipeConnector::set_pipe(shared_ptr<BasicPipe> pipe0)
{
	reset();
	pipe = pipe0;
}

BasicPipe * PipeConnector::get()
{
	return pipe.get();
}

}
}
