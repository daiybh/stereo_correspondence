/*
 * GPUJpegEncoder.h
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#ifndef GPUJPEGENCODER_H_
#define GPUJPEGENCODER_H_
#include "GPUJpegBase.h"
namespace yuri {
namespace gpujpeg {

class GPUJpegEncoder: public GPUJpegBase {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual ~GPUJpegEncoder();
	virtual bool step();
	virtual bool set_param(Parameter &parameter);
protected:
	GPUJpegEncoder(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	bool init();
	bool init_image_params(shared_ptr<BasicFrame> frame);
	gpujpeg_parameters encoder_params;
	gpujpeg_image_parameters image_params;
	shared_ptr<gpujpeg_encoder> encoder;
	yuri::size_t width, height;
	yuri::uint_t quality;
};

} /* namespace io */
} /* namespace yuri */
#endif /* GPUJPEGENCODER_H_ */
