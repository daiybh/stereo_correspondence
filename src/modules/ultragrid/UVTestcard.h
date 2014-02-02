/*!
 * @file 		UVTestcard.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVTESTCARD_H_
#define UVTESTCARD_H_

#include "yuri/ultragrid/UVVideoSource.h"

namespace yuri {
namespace uv_testcard {

class UVTestcard: public ultragrid::UVVideoSource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVTestcard(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVTestcard() noexcept;
private:
	
	virtual bool set_param(const core::Parameter& param) override;
	resolution_t resolution_;
	format_t format_;
	int	fps_;
};

} /* namespace uv_testcard */
} /* namespace yuri */
#endif /* UVTESTCARD_H_ */
