/*
 * Rotate.h
 *
 *  Created on: 9.4.2013
 *      Author: neneko
 */

#ifndef ROTATE_H_
#define ROTATE_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace rotate {

class Rotate: public core::BasicIOThread
{
private:
	Rotate(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter &param);
	size_t 		angle_;
public:
	virtual ~Rotate();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
};

} /* namespace rotate */
} /* namespace yuri */

#endif /* ROTATE_H_ */
