/*
 * MergeFrames.h
 *
 *  Created on: 23.2.2013
 *      Author: neneko
 */

#ifndef SPLITFRAMES_H_
#define SPLITFRAMES_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace split {

class MergeFrames: public core::BasicIOThread
{
private:
	MergeFrames(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter &param);
public:
	virtual ~MergeFrames();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	size_t inputs_;
	size_t current_input_;
};

} /* namespace pass */
} /* namespace yuri */

#endif /* SPLITFRAMES_H_ */
