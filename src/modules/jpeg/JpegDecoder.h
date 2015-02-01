/*!
 * @file 		JpegDecoder.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
	virtual core::pFrame do_special_single_step(core::pCompressedVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;

	bool fast_;
	format_t output_format_;
};

} /* namespace jpeg */
} /* namespace yuri */
#endif /* JPEGDECODER_H_ */
