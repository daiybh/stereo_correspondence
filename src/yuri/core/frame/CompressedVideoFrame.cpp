/*
 * CompressedVideoFrame.cpp
 *
 *  Created on: 4.10.2013
 *      Author: neneko
 */

#include "CompressedVideoFrame.h"
namespace yuri {
namespace core {

CompressedVideoFrame::CompressedVideoFrame(format_t format, resolution_t resolution)
:VideoFrame(format, resolution)
{

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


