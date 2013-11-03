/*
 * BasicIOFilter.cpp
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#include "IOFilter.h"
#include "Convert.h"
#include <cassert>

namespace yuri {
namespace core {

Parameters IOFilter::configure()
{
	return MultiIOFilter::configure();
}

IOFilter::IOFilter(const log::Log &log_, pwThreadBase parent, const std::string& id)
:MultiIOFilter(log_, parent, 1, 1, id),priority_supported_(false)
{

}

IOFilter::~IOFilter() noexcept
{

}

pFrame	IOFilter::simple_single_step(const pFrame& frame)
{
	return do_simple_single_step(frame);
}

std::vector<pFrame> IOFilter::do_single_step(const std::vector<pFrame>& frames)
{
	if (!supported_formats_.empty() && !converter_) {
		converter_.reset(new Convert(log, get_this_ptr(), Convert::configure()));
		add_child(converter_);
	}
	assert (frames.size() == 1 && frames[0]);
	pFrame outframe;
	if (supported_formats_.empty()) {
		outframe = simple_single_step(frames[0]);
	} else {
		pFrame frame;
		if (priority_supported_) frame = converter_->convert_to_any(frames[0], supported_formats_);
		else frame = converter_->convert_to_cheapest(frames[0], supported_formats_);
		if (frame) outframe = simple_single_step(frame);
	}
	if (outframe) return {outframe};
	return {};
}
void IOFilter::set_supported_formats(const std::vector<format_t>& formats)
{
	supported_formats_=formats;
}
void IOFilter::set_supported_priority(bool s)
{
	priority_supported_ = s;
}
}
}


