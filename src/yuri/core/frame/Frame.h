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
typedef shared_ptr<Frame> pFrame;

class Frame: public std::enable_shared_from_this<Frame> {
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
	EXPORT pFrame	get_copy() const;

	/*!
	 * Returns size of data in frame in bytes
	 * @return size of data
	 */
	EXPORT size_t	get_size() const noexcept;
	/*!
	 * Returns timestamp associated with the
	 * @return current timestamp
	 */
	EXPORT timestamp_t		
					get_timestamp() const { return timestamp_; }
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
					get_duration() const { return duration_; }
	/*!
	 * Sets duration for current frame
	 * @param duration duration to set
	 */
	EXPORT void		set_duration(duration_t duration);
	/*!
	 * Returns format of the frame
	 * @return format
	 */
	EXPORT format_t	get_format() const { return format_; }
	/*!
	 * Sets format for the frame
	 * @param format foramt to set
	 */
	EXPORT void		set_format(format_t timestamp);
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
	 * Queries the shared state for being unique
 	 * If the method returns true that subsequent call to get_unique should not copy.
 	 *
	 * @return true if this is the only copy of the frame.
	 */
	EXPORT bool			is_unique() const;
	/*!
	 * Method return a modifiable version of the frame.
	 *
	 * If this frame is shared between more threads, the method returns a copy.
	 * Otherwise return the original frame.
	 * @return A version of this frame that is unique and can be directly modified.
	 */
	EXPORT pFrame 	get_unique();
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
	format_t		format_;
	timestamp_t		timestamp_;
	duration_t		duration_;
	std::string		format_name_;
};

}
}


#endif /* FRAME_H_ */
