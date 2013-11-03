/*!
 * @file 		JpegEncoder.h
 * @author 		<Your name>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef JPEGENCODER_H_
#define JPEGENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
#include "yuri/core/thread/Convert.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri {
namespace jpeg {

class JpegEncoder: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~JpegEncoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param);

	size_t quality_;
};

} /* namespace jpeg */
} /* namespace yuri */
#endif /* JPEGENCODER_H_ */
