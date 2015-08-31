/*!
 * @file 		ConvertPlanes.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		30.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef CONVERTPLANES_H_
#define CONVERTPLANES_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace convert_planar {

class ConvertPlanes: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ConvertPlanes(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ConvertPlanes() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param) override;
	format_t	format_;
};

} /* namespace convert_planar */
} /* namespace yuri */
#endif /* CONVERTPLANES_H_ */
