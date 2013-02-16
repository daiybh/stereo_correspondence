/*!
 * @file 		Crop.h
 * @author 		Zdenek Travnicek
 * @date 		17.11.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef CROP_H_
#define CROP_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"


namespace yuri {

namespace io {
using namespace yuri::config;

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
