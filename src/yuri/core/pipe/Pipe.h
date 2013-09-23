/*
 * BasicPipe.h
 *
 *  Created on: 25.11.2012
 *      Author: neneko
 */

#ifndef BASICPIPE_H_
#define BASICPIPE_H_

#include <atomic>
#include "yuri/core/frame/Frame.h"
#include "yuri/core/pipe/PipeNotification.h"
#include "yuri/log/Log.h"
namespace yuri {
namespace core {
using pPipe = shared_ptr<class Pipe>;

class Pipe
{
public:
	virtual 					~Pipe() noexcept;
	/*!
	 * Push frame 		into the pipe
	 * @param frame 	Frame to push
	 * @return	true is successfully pushed, false if the frame can't be pushed (when pipe is full or closed)
	 */
	bool 						push_frame(const pFrame &frame);
	/*!
	 * Pops a frame out of the pipe
	 * @return	frame or empty pointer
	 */
	pFrame 						pop_frame();
	/*!
	 * Closes this pipe, so no further frames can't be pushed there
	 */
	void						close_pipe();
	/*!
	 * Returns whether the pipe is already finished
	 * @return true when the pipe is closed and empty (i.e. no frame can ever be popped out)
	 */
	bool						is_finished() const;
	/*!
	 * Return number of frames in the pipe
	 * @return Number of pipes in the pipe
	 */
	size_t						get_size() const;
	/*!
	 * Returns whether the pipe is empty
	 * @return true if there's no frame in the pipe
	 */
	bool						is_empty() const;

	void						set_notifiable(pwPipeNotifiable) noexcept;
protected:
								Pipe(const std::string& name, const log::Log& log_);
	log::Log					log;
private:
	virtual bool 				do_push_frame(const pFrame &frame) = 0;
	virtual pFrame 				do_pop_frame() = 0;
	virtual size_t				do_get_size() const = 0;
	void						notify();
	mutex 						frame_lock_;
	std::string 				name_;
	mutable std::atomic<bool>	finished_;
	std::atomic<bool>			closed_;
	pwPipeNotifiable			notifiable_;
	size_t						frames_passed_;
};

} /* namespace core */
} /* namespace yuri */
#endif /* BASICPIPE_H_ */
