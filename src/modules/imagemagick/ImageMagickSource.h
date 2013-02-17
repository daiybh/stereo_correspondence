/*!
 * @file 		DummyModule.h
 * @author 		Zdenek Travnicek
 * @date		17.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/io/BasicIOThread.h"

namespace yuri {
namespace imagemagick_module {
using yuri::log::Log;
using yuri::config::Parameter;
using yuri::config::Parameters;
using yuri::io::pThreadBase;

class ImageMagickSource: public yuri::io::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual ~ImageMagickSource();
private:
	ImageMagickSource(Log &log_,pThreadBase parent,Parameters &parameters);
	virtual bool step();
	virtual bool set_param(Parameter& param);
	yuri::format_t format;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
