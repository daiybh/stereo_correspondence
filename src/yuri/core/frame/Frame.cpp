/*!
 * @file 		Frame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		8.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Frame.h"

namespace yuri {
namespace core {

Frame::Frame(format_t format):format_(format),index_(0)
{

}

Frame::~Frame() noexcept
{

}

void Frame::set_format(format_t format)
{
	format_ = format;
}
void Frame::set_index(index_t index)
{
	index_ = index;
}
void Frame::set_timestamp(timestamp_t timestamp)
{
	timestamp_ = timestamp;
}
void Frame::set_duration(duration_t duration)
{
	duration_ = duration;
}
void Frame::set_format_name(const std::string& format_name)
{
	format_name_ = format_name;
}

void Frame::copy_basic_params(const Frame &other)
{
	set_index(other.get_index());
	set_timestamp(other.get_timestamp());
	set_duration(other.get_duration());
	set_format_name(other.get_format_name());

}
void Frame::copy_parameters(Frame& frame) const
{
	frame.set_format(format_);
	frame.copy_basic_params(*this);
}

}
}


