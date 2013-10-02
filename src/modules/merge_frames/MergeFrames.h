/*
 * MergeFrames.h
 *
 *  Created on: 23.2.2013
 *      Author: neneko
 */

#ifndef SPLITFRAMES_H_
#define SPLITFRAMES_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace split {

class MergeFrames: public core::IOThread
{
public:
	MergeFrames(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~MergeFrames() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	//virtual std::vector<pFrame> do_single_step(const std::vector<core::pFrame>& frames) override;
	virtual bool step() ;
	virtual bool set_param(const core::Parameter &param);
	size_t inputs_;
	size_t current_input_;
};

} /* namespace pass */
} /* namespace yuri */

#endif /* SPLITFRAMES_H_ */
