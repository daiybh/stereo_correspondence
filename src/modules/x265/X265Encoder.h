/*!
 * @file 		X265Encoder.h
 * @author 		<Your name>
 * @date 		30.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef X265ENCODER_H_
#define X265ENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
//#include "yuri/core/thread/ConverterThread.h"
extern "C" {
#include <x265.h>
}

namespace yuri {
namespace x265 {

class X265Encoder:  public core::SpecializedIOFilter<core::RawVideoFrame>//, public core::ConverterThread
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	X265Encoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~X265Encoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;

	void process_nal(x265_nal& nal);
	x265_param params_;
	x265_picture* picture_in_;
	x265_picture* picture_out_;
//	x265_picture picture_out_;
	//	x264_nal_t* nals_;
	x265_encoder* encoder_;
	size_t frame_number_;

	std::string preset_;
	std::string tune_;
	std::string profile_;
};

} /* namespace x265 */
} /* namespace yuri */
#endif /* X265ENCODER_H_ */
