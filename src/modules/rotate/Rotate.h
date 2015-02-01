/*
 * Rotate.h
 *
 *  Created on: 9.4.2013
 *      Author: neneko
 */

#ifndef ROTATE_H_
#define ROTATE_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace rotate {

class Rotate: public core::SpecializedIOFilter<core::RawVideoFrame>
{
public:
	Rotate(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Rotate() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	virtual bool set_param(const core::Parameter &param);
	virtual core::pFrame			do_special_single_step(core::pRawVideoFrame frame) override;
	size_t 		angle_;
};

} /* namespace rotate */
} /* namespace yuri */

#endif /* ROTATE_H_ */
