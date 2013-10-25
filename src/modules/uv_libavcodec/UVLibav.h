/*!
 * @file 		UVLibav.h
 * @author 		<Your name>
 * @date 		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVRTDXT_H_
#define UVRTDXT_H_

#include "yuri/ultragrid/UVVideoCompress.h"
namespace yuri {
namespace uv_rtdxt {



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

} /* namespace uv_rtdxt */
} /* namespace yuri */
#endif /* UVRTDXT_H_ */
