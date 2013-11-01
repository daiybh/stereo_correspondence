/*!
 * @file 		MjpegToJpeg.h
 * @author 		<Your name>
 * @date 		01.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef MJPEGTOJPEG_H_
#define MJPEGTOJPEG_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"

namespace yuri {
namespace mjpeg_to_jpeg {

class MjpegToJpeg: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	MjpegToJpeg(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~MjpegToJpeg() noexcept;
private:
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual core::pFrame do_special_single_step(const core::pCompressedVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace mjpeg_to_jpeg */
} /* namespace yuri */
#endif /* MJPEGTOJPEG_H_ */
