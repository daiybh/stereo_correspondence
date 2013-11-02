/*!
 * @file 		PngEncoder.h
 * @author 		<Your name>
 * @date 		02.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef PNGENCODER_H_
#define PNGENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
#include "yuri/core/thread/Convert.h"
namespace yuri {
namespace png {

class PngEncoder: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	PngEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~PngEncoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param);
	core::pConvert converter_;
};

} /* namespace png */
} /* namespace yuri */
#endif /* PNGENCODER_H_ */
