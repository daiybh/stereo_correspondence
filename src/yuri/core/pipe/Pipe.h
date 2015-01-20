/*!
 * @file 		Pipe.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		25.11.2012
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
	/*!
	 * Returns whether the pipe is full
	 * @return true if there's no frame in the pipe
	 */
	bool						is_full() const noexcept { return do_is_full();};
	void						set_notifiable(pwPipeNotifiable) noexcept;
	void						set_notifiable_source(pwPipeNotifiable) noexcept;

	bool						is_blocking() const noexcept { return do_is_blocking(); }
protected:
								Pipe(const std::string& name, const log::Log& log_);
	void						drop_frame(const pFrame &frame) { if(frame) frames_dropped_++; }
	log::Log					log;
private:
	virtual bool 				do_push_frame(const pFrame &frame) = 0;
	virtual pFrame 				do_pop_frame() = 0;
	virtual size_t				do_get_size() const = 0;
	virtual bool				do_is_full() const noexcept = 0;
	void						notify();
	void						notify_source();
	virtual bool				do_is_blocking() const noexcept = 0;
	mutex 						frame_lock_;
	std::string 				name_;
	mutable std::atomic<bool>	finished_;
	std::atomic<bool>			closed_;
	pwPipeNotifiable			notifiable_;
	pwPipeNotifiable			notifiable_source_;
	size_t						frames_passed_;
	size_t						frames_dropped_;
};

} /* namespace core */
} /* namespace yuri */
#endif /* BASICPIPE_H_ */
