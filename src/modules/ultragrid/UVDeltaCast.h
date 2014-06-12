/*!
 * @file 		UVDeltaCast.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		12.06.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVDELTA_H_
#define UVDELTA_H_

#include "UVVideoSource.h"

namespace yuri {
namespace uv_deltacast {

class UVDeltaCast: public ultragrid::UVVideoSource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVDeltaCast(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVDeltaCast() noexcept;
private:
	virtual bool set_param(const core::Parameter& param) override;
	int	fps_;
};

} /* namespace uv_screen */
} /* namespace yuri */
#endif /* UVDELTA_H_ */
