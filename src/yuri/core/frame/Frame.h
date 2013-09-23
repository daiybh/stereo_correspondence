/*
 * Frame.h
 *
 *  Created on: 30.7.2013
 *      Author: neneko
 */

#ifndef FRAME_H_
#define FRAME_H_
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/Timer.h"
namespace yuri {
namespace core {

class Frame;
typedef shared_ptr<Frame> pFrame;

EXPORT class Frame {
public:
					Frame(format_t format);
	virtual 		~Frame() noexcept;

	/*!
	 * Copy constructor is deleted
	 */
					Frame(const Frame&) 	= delete;
	/*!
	 * Move constructor is deleted
	 */
					Frame(Frame&&) 			= delete;
	/*!
	 * Assignment operator is deleted
	 */
	void 			operator=(const Frame&)	= delete;
	/*!
	 * Move operator is deleted
	 */
	void 			operator=(Frame&&) 		= delete;

	/*!
	 * @brief Returns a deep copy of the frame
	 * @return pointer to the newly created copy
	 */
	pFrame			get_copy() const;

	/*!
	 * Returns size of data in frame in bytes
	 * @return size of data
	 */
	size_t			get_size() const noexcept;
	/*!
	 * Returns timestamp associated with the
	 * @return current timestamp
	 */
	timestamp_t		get_timestamp() const { return timestamp_; }
	/*!
	 * Sets timestamp for the frame
	 * @param timestamp timestamp to set
	 */
	void			set_timestamp(timestamp_t timestamp);
	/*!
	 * Return duration of the frame. Can return duration of 0.
	 * @return frame duration
	 */
	duration_t		get_duration() const { return duration_; }
	/*!
	 * Sets duration for current frame
	 * @param duration duration to set
	 */
	void			set_duration(duration_t duration);
	/*!
	 * Returns format of the frame
	 * @return format
	 */
	format_t		get_format() const { return format_; }
	/*!
	 * Sets format for the frame
	 * @param format foramt to set
	 */
	void			set_format(format_t timestamp);
	/*!
	 * Returns format name of the frame
	 * @return format name
	 */
	std::string		get_format_name() const { return format_name_; }
	/*!
	 * Sets format for the frame
	 * @param format format name to set
	 */
	void			set_format_name(const std::string& format_name);

private:
	/*!
	 * Implementation of copy, should be implemented in node classes only.
	 * @return Copy of current frame
	 */
	virtual pFrame	do_get_copy() const = 0;
	/*!
	 * Implementation od get_size() method
	 * @return Size of current frame
	 */
	virtual size_t	do_get_size() const noexcept = 0;
protected:
	/*!
	 * Copies parameters from the frame to other frame.
	 * The intention is to be used in get_copy() method.
	 * @param frame to copy parameters to
	 */
	virtual void	copy_parameters(Frame&) const;
private:
	format_t		format_;
	timestamp_t		timestamp_;
	duration_t		duration_;
	std::string		format_name_;
};

}
}


#endif /* FRAME_H_ */
