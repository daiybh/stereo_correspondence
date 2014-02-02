/*!
 * @file 		UVScreen.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVSCREEN_H_
#define UVSCREEN_H_

#include "UVVideoSource.h"

namespace yuri {
namespace uv_screen {

class UVScreen: public ultragrid::UVVideoSource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVScreen(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVScreen() noexcept;
private:
	virtual bool set_param(const core::Parameter& param) override;
	int	fps_;
};

} /* namespace uv_screen */
} /* namespace yuri */
#endif /* UVSCREEN_H_ */
