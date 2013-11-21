/*!
 * @file 		PipeConnector.cpp
 * @author 		Zdenek Travnicek
 * @date 		8.8.2010
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PipeConnector.h"
#include "yuri/core/pipe/Pipe.h"
namespace yuri {

namespace core {

PipeConnector::PipeConnector(pwPipeNotifiable thread)
	:notifiable_(thread)
{

}

PipeConnector::PipeConnector(pPipe pipe, pwPipeNotifiable thread)
	:notifiable_(thread)
{
	set_pipe(pipe);
}

PipeConnector::PipeConnector(const PipeConnector& orig):
		notifiable_(orig.notifiable_),pipe_(orig.pipe_)
{

}
PipeConnector::~PipeConnector() noexcept {
	set_notifications(pwPipeNotifiable());
}

pPipe PipeConnector::operator ->()
{
	return pipe_;
}

void PipeConnector::reset()
{
	set_notifications(pwPipeNotifiable());
	pipe_.reset();
}

PipeConnector::operator bool() const
{
	return bool(get());
}

PipeConnector::operator pPipe()
{
	return pipe_;
}

void PipeConnector::reset(pPipe pipe)
{
	set_pipe(pipe);
}

void PipeConnector::set_pipe(pPipe pipe)
{
	set_notifications(pwPipeNotifiable());
	pipe_ = pipe;
	set_notifications(notifiable_);
}
void PipeConnector::set_notifications(pwPipeNotifiable notifiable) noexcept
{
	if (pipe_) pipe_->set_notifiable(notifiable);
}
pPipe PipeConnector::get() const
{
	return pipe_;
}

}
}
