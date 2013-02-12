/*
 * SageOutput.h
 *
 *  Created on: Jan 23, 2013
 *      Author: neneko
 */

#ifndef SAGEOUTPUT_H_
#define SAGEOUTPUT_H_

#include "yuri/io/BasicIOThread.h"
// Libsail has some warnings, let's disable them for the moment (before upstream fixes them)
#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-pedantic"
#endif
#include "libsage.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined __clang__
#pragma clang diagnostic pop
#endif

namespace yuri {
namespace sage {

class SageOutput: public yuri::io::BasicIOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<yuri::config::Parameters> configure();
	virtual ~SageOutput();
private:
	SageOutput(yuri::log::Log &log_,yuri::io::pThreadBase parent,yuri::config::Parameters &p);
	sail *sail_info;
	yuri::size_t width, height;
	yuri::format_t fmt;
	sagePixFmt sage_fmt;
	std::string sage_address;

	virtual bool set_param(yuri::config::Parameter &parameter);
	virtual bool step();
};

} /* namespace sage */
} /* namespace yuri */
#endif /* SAGEOUTPUT_H_ */
