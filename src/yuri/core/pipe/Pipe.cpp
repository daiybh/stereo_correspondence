/*
 * BasicPipe.cpp
 *
 *  Created on: 25.11.2012
 *      Author: neneko
 */

#include "Pipe.h"

namespace yuri {
namespace core {


Pipe::Pipe(const std::string& name, const log::Log& log_):log(log_),name_(name),
		finished_(false),closed_(false),frames_passed_(0),frames_dropped_(0)
{
	log.set_label("[Pipe: "+name+"] ");
}

Pipe::~Pipe() noexcept
{
	log[log::info] << "Processed " << frames_passed_ << " frames, " << frames_dropped_ << " dropped.";
}

pFrame Pipe::pop_frame()
{
	lock_t _(frame_lock_);
	pFrame f = do_pop_frame();
	if (f) frames_passed_++;
	return f;
}

bool Pipe::push_frame(const pFrame &frame)
{
	lock_t _(frame_lock_);
	if (!closed_ && do_push_frame(frame)) {
		notify();
		return true;
	}
	return false;

}

void Pipe::close_pipe()
{
	closed_ = true;
	finished_ = get_size() == 0;
}
bool Pipe::is_finished() const
{
	if (finished_) return true;
	if (closed_ && get_size() == 0) finished_ = true;
	return finished_;
}
size_t Pipe::get_size() const {
	return do_get_size();
}

void Pipe::notify() {
	if (!notifiable_.expired()) {
		auto n = notifiable_.lock();
		n->notify();
	}
}
bool Pipe::is_empty() const {
	return get_size() == 0;
}
void Pipe::set_notifiable(pwPipeNotifiable notifiable) noexcept
{
	notifiable_ = notifiable;
}
} /* namespace core */
} /* namespace yuri */
