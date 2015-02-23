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

PipeConnector::PipeConnector(pwPipeNotifiable thread, pwPipeNotifiable thread_src)
	:notifiable_(thread),notifiable_src_(thread_src)
{
}

PipeConnector::PipeConnector(pPipe pipe, pwPipeNotifiable thread, pwPipeNotifiable thread_src)
	:notifiable_(thread),notifiable_src_(thread_src)
{
	set_pipe(pipe);
}

PipeConnector::PipeConnector(PipeConnector&& rhs) noexcept:
		notifiable_(std::move(rhs.notifiable_)),notifiable_src_(std::move(rhs.notifiable_src_)),
		pipe_(std::move(rhs.pipe_))
{
	rhs.pipe_={};
}
PipeConnector::~PipeConnector() noexcept {
	set_notifications({}, {});
}

PipeConnector&	PipeConnector::operator=(PipeConnector&& rhs) noexcept
{
	// Disconnect notifications if needed
	if (pipe_ != rhs.pipe_) set_notifications({}, {});
	pipe_ = std::move(rhs.pipe_);
	notifiable_=std::move(rhs.notifiable_);
	notifiable_src_=std::move(rhs.notifiable_src_);
	rhs.pipe_={};
	return *this;
}

pPipe PipeConnector::operator ->()
{
	return pipe_;
}

void PipeConnector::reset()
{
	set_notifications({}, {});
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
	set_notifications({}, {});
	pipe_ = pipe;
	set_notifications(notifiable_, notifiable_src_);
}
void PipeConnector::set_notifications(pwPipeNotifiable notifiable, pwPipeNotifiable notifiable_src) noexcept
{
	if (pipe_ && !notifiable_.expired()) pipe_->set_notifiable(notifiable);
	if (pipe_ && !notifiable_src_.expired()) pipe_->set_notifiable_source(notifiable_src);
}
pPipe PipeConnector::get() const
{
	return pipe_;
}

}
}
