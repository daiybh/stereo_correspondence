/*
 * PipePolicies.h
 *
 *  Created on: 25.11.2012
 *      Author: neneko
 */

#ifndef PIPEPOLICIES_H_
#define PIPEPOLICIES_H_
#include "yuri/core/utils/new_types.h"
#include "yuri/core/frame/Frame.h"
#include "yuri/core/parameter/Parameters.h"
#include <deque>

namespace yuri {
namespace core {

namespace pipe {

/*!
 * \brief Policy for pipes without any storage limit
 */
template<bool blocking>
class UnlimitedPolicy {
public:
	static Parameters configure() {return Parameters{};}
protected:
	UnlimitedPolicy(const Parameters&) {}
	~UnlimitedPolicy() noexcept {}
	bool impl_push_frame(const pFrame &frame)
	{
		frames_.push_back(frame);
		return true;
	}
	virtual pFrame impl_pop_frame()
	{
		pFrame frame;
		if (frames_.empty()) return frame;
		frame = frames_.front();
		frames_.pop_front();
		return frame;
	}
	size_t impl_get_size() const {
		return frames_.size();
	}
private:
	std::deque<pFrame> frames_;
};

/*!
 * Policy for pipes able to hold only a single frame
 */

template<bool blocking>
class SingleFramePolicy {
public:
	static Parameters configure() {return Parameters{};}
protected:
	SingleFramePolicy(const Parameters&) {}
	~SingleFramePolicy() noexcept {}
	bool impl_push_frame(const pFrame &frame);
	pFrame impl_pop_frame()
	{
		pFrame frame = frame_;
		frame_.reset();
		return frame;
	}
	size_t impl_get_size() const {
		return frame_?1:0;
	}
private:
	pFrame frame_;
};

/*!
 * Policy for pipes able to store limited number of frames
 */
template<bool blocking>
class SizeLimitedPolicy {
public:
	static Parameters configure() {
			Parameters p;
			p["size"]=1048576;
			return p;
		}
protected:
	SizeLimitedPolicy(const Parameters& parameters):actual_size_(0),max_size_(0)
	{
		/*! @TODO: Process parameters */
		max_size_= parameters["size"].get<size_t>();
	}
	virtual ~SizeLimitedPolicy() noexcept {}
	void set_size_limit(yuri::size_t max_size)
	{
		max_size_ = max_size;
	}
	bool impl_push_frame(const pFrame &frame);
	pFrame impl_pop_frame()
	{
		pFrame frame;
		if (frames_.empty()) return frame;
		frame = frames_.front();
		frames_.pop_front();
		actual_size_-= frame->get_size();
		return frame;
	}
	size_t impl_get_size() const {
		return frames_.size();
	}
private:
	std::deque<pFrame> frames_;
	yuri::size_t actual_size_;
	yuri::size_t max_size_;
};

/*!
 * @brief Policy for pipes able to store only limited amount of data
 */
template<bool blocking>
class CountLimitedPolicy {
public:
	static Parameters configure() {
		Parameters p;
		p["count"]=10;
		return p;
	}
protected:
	CountLimitedPolicy(const Parameters& parameters):max_count_(0)
	{
		max_count_=parameters["count"].get<size_t>();
	}
	virtual ~CountLimitedPolicy() noexcept {}
	void set_count_limit(yuri::size_t max_count)
	{
		max_count_ = max_count;
	}
	bool impl_push_frame(const pFrame &frame);
	pFrame impl_pop_frame()
	{
		pFrame frame;
		if (frames_.empty()) return frame;
		frame = frames_.front();
		frames_.pop_front();
		return frame;
	}
	size_t impl_get_size() const {
		return frames_.size();
	}
private:
	std::deque<pFrame> frames_;
	yuri::size_t max_count_;
};


}

} /* namespace core */
} /* namespace yuri */
#endif /* PIPEPOLICIES_H_ */
