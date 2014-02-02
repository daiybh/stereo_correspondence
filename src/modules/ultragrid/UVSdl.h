/*!
 * @file 		UVSdl.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVSDL_H_
#define UVSDL_H_

#include "UVVideoSink.h"
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
