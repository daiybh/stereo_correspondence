/*
 * Flip.h
 *
 *  Created on: Mar 16, 2012
 *      Author: worker
 */

#ifndef FLIP_H_
#define FLIP_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"

namespace yuri {

namespace io {

class Flip: public yuri::io::BasicIOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual bool set_param(Parameter &parameter);

	virtual ~Flip();
protected:
	Flip(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual bool step();

	bool flip_x, flip_y;
};

}

}

#endif /* FLIP_H_ */
