/*
 * GPUJpegEncoder.h
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#ifndef GPUJPEGENCODER_H_
#define GPUJPEGENCODER_H_
#include "GPUJpegBase.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace gpujpeg {

class GPUJpegEncoder: public GPUJpegBase, public core::ConverterThread {
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	GPUJpegEncoder(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~GPUJpegEncoder() noexcept;
	virtual bool set_param(const core::Parameter &parameter);
protected:

	bool init();
	bool init_image_params(const core::pRawVideoFrame& frame);
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool do_initialize_converter(format_t target_format) override; //{ return true; }
	virtual bool do_converter_is_stateless() const override { return false; }
	gpujpeg_parameters encoder_params;
	gpujpeg_image_parameters image_params;
	std::shared_ptr<gpujpeg_encoder> encoder;
	size_t width, height;
	uint16_t quality;
};

} /* namespace io */
} /* namespace yuri */
#endif /* GPUJPEGENCODER_H_ */
