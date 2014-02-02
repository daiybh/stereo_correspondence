/*!
 * @file 		UVUyvy.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVUYVY_H_
#define UVUYVY_H_

#include "UVVideoCompress.h"

namespace yuri {
namespace uv_uyvy {

class UVUyvy: public ultragrid::UVVideoCompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVUyvy(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVUyvy() noexcept;
private:
	
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace uv_uyvy */
} /* namespace yuri */
#endif /* UVUYVY_H_ */
