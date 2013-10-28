/*!
 * @file 		BlankGenerator.h
 * @author 		Zdenek Travnicek
 * @date 		11.9.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BLANKGENERATOR_H_
#define BLANKGENERATOR_H_

#include "yuri/core/IOThread.h"
//#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace blank {
//using namespace boost::posix_time;

class BlankGenerator: public core::IOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	BlankGenerator(log::Log &log_,core::pwThreadBase parent,core::Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~BlankGenerator();
	void run();
	bool set_param(const core::Parameter &p);
	core::pBasicFrame  generate_frame();
protected:
	core::pBasicFrame  generate_frame_yuv422();
	core::pBasicFrame  generate_frame_rgb();
	std::map<yuri::format_t,core::pBasicFrame  > blank_frames;
	time_value next_time;
	float fps;
	yuri::ushort_t width;
	yuri::ushort_t height;
	yuri::format_t format;
};

}

}

#endif /* BLANKGENERATOR_H_ */
