/*!
 * @file 		UVLibav.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVLIBAV_H_
#define UVLIBAV_H_

#include "yuri/ultragrid/UVVideoCompress.h"

namespace yuri {
namespace uv_libav {



class UVLibav: public ultragrid::UVVideoCompress
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVLibav(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVLibav() noexcept;
private:
	
	virtual bool 				set_param(const core::Parameter& param);
	format_t 					format_;
	ssize_t						bps_;
	std::string 				subsampling_;
	std::string					preset_;
};

} /* namespace uv_libav*/
} /* namespace yuri */
#endif /* UVLIBAV_H_ */
