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

#include "yuri/core/BasicIOThread.h"

namespace yuri {

namespace io {

class Crop: public core::BasicIOThread {
public:
	virtual ~Crop();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual bool set_param(const core::Parameter &parameter);
protected:
	Crop(log::Log &_log, core::pwThreadBase parent,core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual bool step();
	yuri::ssize_t dest_x;
	yuri::ssize_t dest_y;
	yuri::ssize_t dest_w;
	yuri::ssize_t dest_h;
};

}

}

#endif /* CROP_H_ */
