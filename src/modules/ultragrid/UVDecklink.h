/*!
 * @file 		UVDecklink.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.06.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVDECKLINK_H_
#define UVDECKLINK_H_

#include "UVVideoSource.h"

namespace yuri {
namespace uv_decklink {

class UVDecklink: public ultragrid::UVVideoSource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVDecklink(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVDecklink() noexcept;
private:
	virtual bool set_param(const core::Parameter& param) override;
	int	device_;
	int mode_;
	std::string connection_;


};

} /* namespace uv_screen */
} /* namespace yuri */
#endif /* UVDECKLINK_H_ */
