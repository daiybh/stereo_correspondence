/*!
 * @file 		EventFrame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENTFRAME_H_
#define EVENTFRAME_H_

#include "Frame.h"
#include "yuri/event/BasicEvent.h"
namespace yuri {
namespace core {

class EventFrame;
using pEventFrame = std::shared_ptr<EventFrame>;

class EventFrame: public Frame
{
public:
	EXPORT EventFrame(std::string name, event::pBasicEvent event);
	EXPORT virtual ~EventFrame() noexcept;

	/*!
	 * Returns resolution associated with the
	 * @return frame resolution
	 */
	EXPORT const std::string&	get_name() const { return name_; }
	/*!
	 * Sets resolution for the frame
	 * @param resolution resolution to set
	 */
	EXPORT void					set_name(std::string name);
	/*!
	 * Returns width of the frame
	 * @return frame width
	 */
	EXPORT const event::pBasicEvent&
								get_event() const { return event_;}
	/*!
	 * Returns height of the frame
	 * @return frame height
	 */
	EXPORT void					set_event(event::pBasicEvent event);
protected:
	/*!
	 * Copies parameters from the frame to other frame.
	 * The intention is to be used in get_copy() method.
	 * @param frame to copy parameters to
	 */
	EXPORT virtual void	copy_parameters(Frame&) const override;
	/*!
	 * Implementation of copy, should be implemented in node classes only.
	 * @return Copy of current frame
	 */
	EXPORT virtual pFrame	do_get_copy() const override;
	/*!
	 * Implementation od get_size() method
	 * @return Size of current frame
	 */
	EXPORT virtual size_t	do_get_size() const noexcept override;
private:
	std::string				name_;
	event::pBasicEvent		event_;
};

}
}





#endif /* EVENTFRAME_H_ */
