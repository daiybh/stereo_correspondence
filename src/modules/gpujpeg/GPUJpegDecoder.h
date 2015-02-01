/*
 * GPUJpegDecoder.h
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#ifndef GPUJPEGDECODER_H_
#define GPUJPEGDECODER_H_
#include "GPUJpegBase.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace gpujpeg {

class GPUJpegDecoder: public GPUJpegBase, public core::ConverterThread {
public:
	IOTHREAD_GENERATOR_DECLARATION
	GPUJpegDecoder(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	static core::Parameters configure();
	virtual ~GPUJpegDecoder() noexcept;


protected:
//	virtual bool step();
	virtual bool set_param(const core::Parameter& parameter) override;
	virtual core::pFrame do_simple_single_step(core::pFrame frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool do_initialize_converter(format_t target_format) override; //{ return true; }
	virtual bool do_converter_is_stateless() const override { return false; }
	bool init();
	std::shared_ptr<gpujpeg_decoder> decoder;
	gpujpeg_decoder_output decoder_output;
	format_t out_format;
	gpujpeg_color_space color_space;
	gpujpeg_sampling_factor sampling;
};

} /* namespace io */
} /* namespace yuri */
#endif /* GPUJPEGDECODER_H_ */
