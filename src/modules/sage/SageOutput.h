/*!
 * @file 		SageOutput.h
 * @author 		Zdenek Travnicek
 * @date 		23.1.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SAGEOUTPUT_H_
#define SAGEOUTPUT_H_

#include "yuri/core/thread/IOFilter.h"
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

class SageOutput: public yuri::core::IOFilter{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual ~SageOutput() noexcept;
	SageOutput(log::Log &log_,core::pwThreadBase parent, const core::Parameters &p);
private:

	sail *sail_info;
	yuri::size_t width, height;
	yuri::format_t fmt;
	sagePixFmt sage_fmt;
	std::string sage_address;
	std::string app_name_;

	virtual bool set_param(const core::Parameter &parameter) override;
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual bool step() override;
};

} /* namespace sage */
} /* namespace yuri */
#endif /* SAGEOUTPUT_H_ */
