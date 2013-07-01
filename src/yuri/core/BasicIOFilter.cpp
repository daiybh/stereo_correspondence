/*
 * BasicIOFilter.cpp
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#include "BasicIOFilter.h"

namespace yuri {
namespace core {

BasicIOFilter::BasicIOFilter(log::Log &log_, pwThreadBase parent,std::string id)
:BasicMultiIOFilter(log_, parent, 1, 1, id) {

}

BasicIOFilter::~BasicIOFilter()
{

}

pBasicFrame	BasicIOFilter::simple_single_step(const pBasicFrame& frame)
{
	return do_simple_single_step(frame);
}

std::vector<pBasicFrame> BasicIOFilter::do_single_step(const std::vector<pBasicFrame>& frames)
{
	assert (frames.size() == 1 && frames[0]);
	const pBasicFrame& frame = frames[0];
	pBasicFrame outframe = simple_single_step(frame);
	if (outframe) return {outframe};
	return {};
}

}
}


