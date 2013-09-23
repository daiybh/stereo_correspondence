/*
 * VideoFrame.h
 *
 *  Created on: 30.7.2013
 *      Author: neneko
 */

#ifndef VIDEOFRAME_H_
#define VIDEOFRAME_H_
#include "Frame.h"

namespace yuri {
namespace core {

class VideoFrame;
typedef shared_ptr<VideoFrame> pVideoFrame;

EXPORT class VideoFrame: public Frame
{
public:
	VideoFrame(format_t format, resolution_t resolution, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);
	virtual ~VideoFrame() noexcept;
	// TODO: Implement getters and setters for private fields;

	/*!
	 * Returns resolution associated with the
	 * @return frame resolution
	 */
	resolution_t	get_resolution() const { return resolution_; }
	/*!
	 * Sets resolution for the frame
	 * @param resolution resolution to set
	 */
	void			set_resolution(resolution_t resolution);
	/*!
	 * Returns interlacing type for the frame
	 * @return Interlacing type
	 */
	interlace_t		get_interlacing() const { return interlacing_; }
	/*!
	 * Sets interlacing type for the frame
	 * @param interlacinf Interlacing type to set
	 */
	void			set_interlacing(interlace_t interlacing);
	/*!
	 * Returns filed order for the frame
	 * @return field order
	 */
	field_order_t	get_field_order() const { return field_order_; }
	/*!
	 * Sets field order for the frame
	 * @param field_order Field order to set
	 */
	void			set_field_order(field_order_t field_order);

protected:
	/*!
	 * Copies parameters from the frame to other frame.
	 * The intention is to be used in get_copy() method.
	 * @param frame to copy parameters to
	 */
	virtual void	copy_parameters(Frame&) const;
private:
	resolution_t	resolution_;
	interlace_t		interlacing_;
	field_order_t	field_order_;
};

}
}



#endif /* VIDEOFRAME_H_ */
