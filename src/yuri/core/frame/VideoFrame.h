/*!
 * @file 		VideoFrame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		30.7.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef VIDEOFRAME_H_
#define VIDEOFRAME_H_
#include "Frame.h"

namespace yuri {
namespace core {

class VideoFrame;
typedef std::shared_ptr<VideoFrame> pVideoFrame;

class VideoFrame: public Frame
{
public:
	EXPORT VideoFrame(format_t format, resolution_t resolution, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);
	EXPORT virtual ~VideoFrame() noexcept;
	// TODO: Implement getters and setters for private fields;

	/*!
	 * Returns resolution associated with the
	 * @return frame resolution
	 */
	EXPORT resolution_t	get_resolution() const { return resolution_; }
	/*!
	 * Sets resolution for the frame
	 * @param resolution resolution to set
	 */
	EXPORT void			set_resolution(resolution_t resolution);
	/*!
	 * Returns width of the frame
	 * @return frame width
	 */
	EXPORT dimension_t		get_width() const { return resolution_.width;}
	/*!
	 * Returns height of the frame
	 * @return frame height
	 */
	EXPORT dimension_t		get_height() const { return resolution_.height;}
	/*!
	 * Returns interlacing type for the frame
	 * @return Interlacing type
	 */
	EXPORT interlace_t		get_interlacing() const { return interlacing_; }
	/*!
	 * Sets interlacing type for the frame
	 * @param interlacinf Interlacing type to set
	 */
	EXPORT void			set_interlacing(interlace_t interlacing);
	/*!
	 * Returns filed order for the frame
	 * @return field order
	 */
	EXPORT field_order_t	get_field_order() const { return field_order_; }
	/*!
	 * Sets field order for the frame
	 * @param field_order Field order to set
	 */
	EXPORT void			set_field_order(field_order_t field_order);
	/*!
	 * Sets essential video info from other frame (interlacing, field_order)
	 * @param other Source frame
	 */
	EXPORT void 	copy_video_params(const VideoFrame &other);

protected:
	/*!
	 * Copies parameters from the frame to other frame.
	 * The intention is to be used in get_copy() method.
	 * @param frame to copy parameters to
	 */
	EXPORT virtual void	copy_parameters(Frame&) const;
private:
	resolution_t	resolution_;
	interlace_t		interlacing_;
	field_order_t	field_order_;
};

}
}



#endif /* VIDEOFRAME_H_ */
