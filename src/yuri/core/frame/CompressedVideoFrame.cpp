/*!
 * @file 		CompressedVideoFrame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		4.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "CompressedVideoFrame.h"
namespace yuri {
namespace core {

CompressedVideoFrame::CompressedVideoFrame(format_t format, resolution_t resolution)
:VideoFrame(format, resolution)
{

}
CompressedVideoFrame::CompressedVideoFrame(format_t format, resolution_t resolution, size_t size)
:VideoFrame(format, resolution)
{
	data_.resize(size);
}

CompressedVideoFrame::CompressedVideoFrame(format_t format, resolution_t resolution, const uint8_t* data, size_t size)
:VideoFrame(format, resolution)
{
	data_.resize(size);
	std::copy(data,data+size,data_.begin());
}

CompressedVideoFrame::~CompressedVideoFrame() noexcept
{

}




}
}


