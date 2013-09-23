/*
 * Frame.cpp
 *
 *  Created on: 8.9.2013
 *      Author: neneko
 */

#include "Frame.h"

namespace yuri {
namespace core {

Frame::Frame(format_t format):format_(format)
{

}

Frame::~Frame() noexcept
{

}

pFrame Frame::get_copy() const {
	return do_get_copy();
}

size_t Frame::get_size() const noexcept {
	return do_get_size();
}

void Frame::set_timestamp(timestamp_t timestamp)
{
	timestamp_ = timestamp;
}
void Frame::set_duration(duration_t duration)
{
	duration_ = duration;
}
void Frame::set_format(format_t format)
{
	format_ = format;
}
void Frame::set_format_name(const std::string& format_name)
{
	format_name_ = format_name;
}


void Frame::copy_parameters(Frame& frame) const
{
	frame.set_format(format_);
	frame.set_timestamp(timestamp_);
	frame.set_duration(duration_);
	frame.set_format_name(format_name_);
}

}
}


