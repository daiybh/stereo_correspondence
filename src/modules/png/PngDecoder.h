/*!
 * @file 		PngDecoder.h
 * @author 		<Your name>
 * @date 		02.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef PNGDECODER_H_
#define PNGDECODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace png {

class PngDecoder: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	PngDecoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~PngDecoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pCompressedVideoFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param);
	format_t requested_format_;
};

} /* namespace png */
} /* namespace yuri */
#endif /* PNGDECODER_H_ */
