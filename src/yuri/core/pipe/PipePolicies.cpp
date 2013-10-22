/*
 * PipePolicies.cpp
 *
 *  Created on: 25.11.2012
 *      Author: neneko
 */

#include "PipePolicies.h"
#include <cassert>

namespace yuri {
namespace core {
namespace pipe {

template<>
bool SingleFramePolicy<false>::impl_push_frame(const pFrame &frame)
{
	drop_frame(frame_);
	frame_ = frame;
	return true;
}
template<>
bool SingleFramePolicy<true>::impl_push_frame(const pFrame &frame)
{
	if (frame_) return false;
	frame_ = frame;
	return true;
}

template<>
bool SizeLimitedPolicy<false>::impl_push_frame(const pFrame &frame)
{
	frames_.push_back(frame);
	actual_size_+=frame->get_size();
	while (actual_size_>max_size_) {
		assert(frames_.size());
		pFrame f = frames_.front();
		frames_.pop_front();
		actual_size_-=f->get_size();
		drop_frame(f);
	}
	return true;
}
template<>
bool SizeLimitedPolicy<true>::impl_push_frame(const pFrame &frame)
{
	if ((actual_size_ + frame->get_size()) > max_size_) return false;
	frames_.push_back(frame);
	actual_size_+=frame->get_size();
	return true;
}

template<>
bool CountLimitedPolicy<false>::impl_push_frame(const pFrame &frame)
{
	frames_.push_back(frame);
	while (frames_.size()>max_count_) {
		drop_frame(frames_.front());
		frames_.pop_front();
	}
	return true;
}

template<>
bool CountLimitedPolicy<true>::impl_push_frame(const pFrame &frame)
{
	if (frames_.size()>=max_count_) return false;
	frames_.push_back(frame);
	return true;
}


}
} /* namespace core */
} /* namespace yuri */
