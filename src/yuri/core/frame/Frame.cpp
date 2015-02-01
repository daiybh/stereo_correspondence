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

bool Frame::is_unique() const
{
	auto frame = shared_from_this();
	// If there's use count 2, we have one copy and the caller has another
	// If there's only 1, we have unique copy and caller has none.
	// Any other number means, it can be shared.
	if (frame.use_count() <= 2) {
		return true;
	}
	return false;
}

pFrame Frame::get_unique()
{
	auto frame = shared_from_this();
	// If there's use count 2, we have one copy and the caller has another
	// If there's only 1, we have unique copy and caller has none.
	// Any other number means, it can be shared.
	if (frame.use_count() <= 2) {
		return frame;
	}
	return get_copy();
}

}
}


