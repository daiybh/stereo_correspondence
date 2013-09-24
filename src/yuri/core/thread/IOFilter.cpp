/*
 * BasicIOFilter.cpp
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#include "IOFilter.h"
#include <cassert>

namespace yuri {
namespace core {

Parameters IOFilter::configure()
{
	return MultiIOFilter::configure();
}

IOFilter::IOFilter(const log::Log &log_, pwThreadBase parent, const std::string& id)
:MultiIOFilter(log_, parent, 1, 1, id) {

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
	assert (frames.size() == 1 && frames[0]);
	const pFrame& frame = frames[0];
	pFrame outframe = simple_single_step(frame);
	if (outframe) return {outframe};
	return {};
}

}
}


