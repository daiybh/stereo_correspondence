/*!
 * @file 		UVV4l2.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVV4L2_H_
#define UVV4L2_H_

#include "UVVideoSource.h"


namespace yuri {
namespace uv_v4l2 {

class UVV4l2: public ultragrid::UVVideoSource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVV4l2(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVV4l2() noexcept;
private:
	
	virtual bool set_param(const core::Parameter& param) override;
	std::string device_;
	int	fps_;
	resolution_t resolution_;
	format_t format_;
};

} /* namespace uv_v4l2 */
} /* namespace yuri */
#endif /* UVV4L2_H_ */
