/*
 * BlankGenerator.h
 *
 *  Created on: Sep 11, 2010
 *      Author: neneko
 */

#ifndef BLANKGENERATOR_H_
#define BLANKGENERATOR_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace io {
using namespace boost::posix_time;

class BlankGenerator: public yuri::io::BasicIOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	BlankGenerator(Log &log_,pThreadBase parent, Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~BlankGenerator();
	void run();
	bool set_param(Parameter &p);
	pBasicFrame generate_frame();
protected:
	pBasicFrame generate_frame_yuv422();
	pBasicFrame generate_frame_rgb();
	std::map<yuri::format_t,pBasicFrame > blank_frames;
	ptime next_time;
	float fps;
	yuri::ushort_t width;
	yuri::ushort_t height;
	yuri::format_t format;
};

}

}

#endif /* BLANKGENERATOR_H_ */
