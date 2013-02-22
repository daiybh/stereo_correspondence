/*!
 * @file 		SageOutput.h
 * @author 		Zdenek Travnicek
 * @date 		23.1.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef SAGEOUTPUT_H_
#define SAGEOUTPUT_H_

#include "yuri/core/BasicIOThread.h"
// Libsail has some warnings, let's disable them for the moment (before upstream fixes them)
// Clang has to be first as it also defines __GNUC__
#if defined __clang__
#pragma clang diagnostic push
// -Wc++11-extra-semi is not supported in clang3.1, so ignore that pragma
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wc++11-extra-semi"
#elif defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-pedantic"
#endif
#include "libsage.h"
#if defined __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__
#pragma GCC  diagnostic pop
#endif

namespace yuri {
namespace sage {

class SageOutput: public yuri::core::BasicIOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~SageOutput();
private:
	SageOutput(log::Log &log_,core::pwThreadBase parent,core::Parameters &p);
	sail *sail_info;
	yuri::size_t width, height;
	yuri::format_t fmt;
	sagePixFmt sage_fmt;
	std::string sage_address;

	virtual bool set_param(const core::Parameter &parameter);
	virtual bool step();
};

} /* namespace sage */
} /* namespace yuri */
#endif /* SAGEOUTPUT_H_ */
