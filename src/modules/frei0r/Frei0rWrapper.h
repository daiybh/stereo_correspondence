/*!
 * @file 		Frei0rWrapper.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FREI0RWRAPPER_H_
#define FREI0RWRAPPER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "Frei0rBase.h"

namespace yuri {
namespace frei0r {

class Frei0rWrapper: public core::SpecializedIOFilter<core::RawVideoFrame>, private Frei0rBase
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;


public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Frei0rWrapper(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Frei0rWrapper() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;


	resolution_t last_res_;
};

} /* namespace frei0r */
} /* namespace yuri */
#endif /* FREI0RWRAPPER_H_ */
