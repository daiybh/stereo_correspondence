/*!
 * @file 		X264Encoder.h
 * @author 		<Your name>
 * @date 		28.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef X264ENCODER_H_
#define X264ENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
//#include "yuri/core/thread/ConverterThread.h"
#ifndef _STDINT_H
// x264 expects stdint.h to be included ot emits a warning. We include <cstdint>, this is simply to prevent that warning.
#define _STDINT_H 1
#endif
extern "C" {
#include <x264.h>
}
namespace yuri {
namespace x264 {

class X264Encoder: public core::SpecializedIOFilter<core::RawVideoFrame>//, public core::ConverterThread
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	X264Encoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~X264Encoder() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;

	void process_nal(x264_nal_t& nal);
	x264_param_t params_;
	x264_picture_t picture_in_;
	x264_picture_t picture_out_;
//	x264_nal_t* nals_;
	x264_t* encoder_;
	size_t frame_number_;

	std::string preset_;
	std::string tune_;
	std::string profile_;

};

} /* namespace x264 */
} /* namespace yuri */
#endif /* X264ENCODER_H_ */
