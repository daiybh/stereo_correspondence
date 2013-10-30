/*!
 * @file 		ConvertPlanes.h
 * @author 		<Your name>
 * @date 		30.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
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
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param);
	format_t	format_;
};

} /* namespace convert_planar */
} /* namespace yuri */
#endif /* CONVERTPLANES_H_ */
