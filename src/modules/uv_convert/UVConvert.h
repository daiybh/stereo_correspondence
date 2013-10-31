/*!
 * @file 		UVConvert.h
 * @author 		<Your name>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVCONVERT_H_
#define UVCONVERT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace uv_convert {

class UVConvert: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVConvert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVConvert() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;

	format_t format_;
};

} /* namespace uv_convert */
} /* namespace yuri */
#endif /* UVCONVERT_H_ */
