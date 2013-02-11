/*
 * Crop.h
 *
 *  Created on: Nov 17, 2010
 *      Author: neneko
 */

#ifndef CROP_H_
#define CROP_H_

#include "BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"


namespace yuri {

namespace io {
using namespace yuri::config;
using boost::shared_array;
class Crop: public yuri::io::BasicIOThread {
public:
	virtual ~Crop();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual bool set_param(Parameter &parameter);
protected:
	Crop(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual bool step();
	yuri::ssize_t dest_x;
	yuri::ssize_t dest_y;
	yuri::ssize_t dest_w;
	yuri::ssize_t dest_h;
};

}

}

#endif /* CROP_H_ */
