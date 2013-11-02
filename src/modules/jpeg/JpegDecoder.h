/*!
 * @file 		JpegDecoder.h
 * @author 		<Your name>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef JPEGDECODER_H_
#define JPEGDECODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"


namespace yuri {
namespace jpeg {

class JpegDecoder: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	JpegDecoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~JpegDecoder() noexcept;

private:
	virtual core::pFrame do_special_single_step(const core::pCompressedVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;

	bool fast_;
	format_t output_format_;
};

} /* namespace jpeg */
} /* namespace yuri */
#endif /* JPEGDECODER_H_ */
