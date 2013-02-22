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

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace imagemagick_module {

class ImageMagickSource: public yuri::core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~ImageMagickSource();
private:
	ImageMagickSource(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	yuri::format_t format;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
