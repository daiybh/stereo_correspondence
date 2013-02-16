/*!
 * @file 		Flip.h
 * @author 		Zdenek Travnicek
 * @date 		16.3.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
 *
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
