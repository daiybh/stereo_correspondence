/*
 * SplitFrames.h
 *
 *  Created on: 23.2.2013
 *      Author: neneko
 */

#ifndef SPLITFRAMES_H_
#define SPLITFRAMES_H_

#include "yuri/core/thread/MultiIOFilter.h"

namespace yuri {
namespace split {

class SplitFrames: public core::MultiIOFilter
{
public:
	SplitFrames(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SplitFrames() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	size_t outputs_;
	size_t current_output_;
private:
	virtual std::vector<core::pFrame> do_single_step(std::vector<core::pFrame> frames) override;
	virtual bool set_param(const core::Parameter &param);
};

} /* namespace pass */
} /* namespace yuri */

#endif /* SPLITFRAMES_H_ */
