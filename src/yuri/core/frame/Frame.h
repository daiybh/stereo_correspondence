/*!
 * @file 		Frame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		30.7.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FRAME_H_
#define FRAME_H_
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/Timer.h"
namespace yuri {
namespace core {

class Frame;
typedef std::shared_ptr<Frame> pFrame;

/*!
 * Queries the shared state for being unique
 * If the method returns true that subsequent call to get_unique should not copy.
 *
 * @return true if this is the only copy of the frame.
 */
template<class T>
typename std::enable_if<std::is_base_of<core::Frame, T>::value, bool>::type
is_frame_unique(const std::shared_ptr<T>& frame) {
	return frame.use_count() == 1;
}

/*!
 * Method returns a modifiable version of the frame.
 *
 * If this frame is shared between more threads, the method returns a copy.
 * Otherwise returns the original frame.
 * @return A version of this frame that is unique and can be directly modified.
 */
template<class T>
typename std::enable_if<std::is_base_of<core::Frame, T>::value, std::shared_ptr<T>>::type
get_frame_unique(const std::shared_ptr<T>& frame)
{
	if (frame && !is_frame_unique(frame)) return std::dynamic_pointer_cast<T>(frame->get_copy());
	return frame;
}


class Frame {
public:
	EXPORT 			Frame(format_t format);
	EXPORT virtual 	~Frame() noexcept;

	/*!
	 * Copy constructor is deleted
	 */
	EXPORT 			Frame(const Frame&) 	= delete;
	/*!
	 * Move constructor is deleted
	 */
	EXPORT 			Frame(Frame&&) 			= delete;
	/*!
	 * Assignment operator is deleted
	 */
	EXPORT void 	operator=(const Frame&)	= delete;
	/*!
	 * Move operator is deleted
	 */
	EXPORT void 	operator=(Frame&&) 		= delete;

	/*!
	 * @brief Returns a deep copy of the frame
	 * @return pointer to the newly created copy
	 */
	EXPORT pFrame	get_copy() const { return do_get_copy(); }

	/*!
	 * Returns size of data in frame in bytes
	 * @return size of data
	 */
	EXPORT size_t	get_size() const noexcept { return do_get_size(); }
	/*!
	 * Returns format of the frame
	 * @return format
	 */
	EXPORT format_t	get_format() const noexcept { return format_; }
	/*!
	 * Sets format for the frame
	 * @param format foramt to set
	 */
	EXPORT void		set_format(format_t timestamp);
	/*!
	 * Returns index of the frame
	 * @return index
	 */
	EXPORT index_t	get_index() const noexcept { return index_; }
	/*!
	 * Sets format for the frame
	 * @param index index to set
	 */
	EXPORT void		set_index(index_t index);
	/*!
	 * Returns timestamp associated with the
	 * @return current timestamp
	 */
	EXPORT timestamp_t		
					get_timestamp() const noexcept { return timestamp_; }
	/*!
	 * Sets timestamp for the frame
	 * @param timestamp timestamp to set
	 */
	EXPORT void		set_timestamp(timestamp_t timestamp);
	/*!
	 * Return duration of the frame. Can return duration of 0.
	 * @return frame duration
	 */
	EXPORT duration_t	
					get_duration() const noexcept { return duration_; }
	/*!
	 * Sets duration for current frame
	 * @param duration duration to set
	 */
	EXPORT void		set_duration(duration_t duration);
	/*!
	 * Returns format name of the frame
	 * @return format name
	 */
	EXPORT std::string		
					get_format_name() const { return format_name_; }
	/*!
	 * Sets format for the frame
	 * @param format format name to set
	 */
	EXPORT void		set_format_name(const std::string& format_name);
	/*!
	 * Sets all timing info (index, timestamp, duration and format_name) from other frame.
	 * @param other Source frame
	 */
	EXPORT void 	copy_basic_params(const Frame &other);
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
	EXPORT 	virtual void	
					copy_parameters(Frame&) const;
private:
	//! Frame format
	format_t		format_;
	//! Frame position the stream
	index_t			index_;
	//! Time the frame was generated
	timestamp_t		timestamp_;
	//! Frame duration (1/fps)
	duration_t		duration_;
	//! An arbitrary string describing the format (candidate for removal, not really used anymore)
	std::string		format_name_;
};

}
}


#endif /* FRAME_H_ */
