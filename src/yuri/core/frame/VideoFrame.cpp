/*!
 * @file 		VideoFrame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		8.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "VideoFrame.h"

namespace yuri {
namespace core {


VideoFrame::VideoFrame(format_t format, resolution_t resolution, interlace_t interlace, field_order_t field_order)
:Frame(format),resolution_(resolution), interlacing_(interlace),field_order_(field_order)
{

}

VideoFrame::~VideoFrame() noexcept
{

}

void VideoFrame::set_resolution(resolution_t resolution)
{
	resolution_ = resolution;
}
void VideoFrame::set_interlacing(interlace_t interlacing)
{
	interlacing_ = interlacing;
}
void VideoFrame::set_field_order(field_order_t field_order)
{
	field_order_ = field_order;
}

void VideoFrame::copy_video_params(const VideoFrame &other)
{
	set_interlacing(other.get_interlacing());
	set_field_order(other.get_field_order());
	copy_basic_params(other);
}

void VideoFrame::copy_parameters(Frame& other) const
{
	try {
		VideoFrame& frame = dynamic_cast<VideoFrame&>(other);
		frame.set_resolution(resolution_);
		frame.copy_video_params(*this);
	}
	catch (std::bad_cast&) {
		throw std::runtime_error("Tried to set VideoFrame params to a type not related to VideoFrame");
	}
	Frame::copy_parameters(other);
}
}
}

