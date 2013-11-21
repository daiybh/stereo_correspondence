/*!
 * @file 		UVGl.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVGL_H_
#define UVGL_H_

#include "yuri/ultragrid/UVVideoSink.h"

namespace yuri {
namespace uv_gl {

class UVGl: public ultragrid::UVVideoSink
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVGl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVGl() noexcept;
private:
	virtual bool set_param(const core::Parameter& param) override;
};

} /* namespace uv_gl */
} /* namespace yuri */
#endif /* UVGL_H_ */
