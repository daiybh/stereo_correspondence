/*!
 * @file 		UVSdl.h
 * @author 		<Your name>
 * @date 		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVSDL_H_
#define UVSDL_H_

#include "yuri/ultragrid/UVVideoSink.h"
namespace yuri {
namespace uv_sdl {

class UVSdl: public ultragrid::UVVideoSink
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVSdl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVSdl() noexcept;
private:
	virtual bool set_param(const core::Parameter& param);
	bool fullscreen_;
	bool deinterlace_;

};

} /* namespace uv_sdl */
} /* namespace yuri */
#endif /* UVSDL_H_ */
