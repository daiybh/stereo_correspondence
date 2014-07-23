/*
 * GPUJpegEncoder.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#include "GPUJpegEncoder.h"

namespace yuri {
namespace gpujpeg {

REGISTER("gpujpeg_encoder",GPUJpegEncoder)
IO_THREAD_GENERATOR(GPUJpegEncoder)
shared_ptr<Parameters> GPUJpegEncoder::configure()
{
	shared_ptr<Parameters> p = GPUJpegBase::configure();
	(*p)["quality"]["Jpeg quality (0-100)"]=90;
	p->set_max_pipes(1,1);
	p->add_input_format(YURI_FMT_YUV422);
	p->add_output_format(YURI_IMAGE_JPEG);
	return p;
}


GPUJpegEncoder::GPUJpegEncoder(Log &_log, pThreadBase parent, Parameters &parameters)
IO_THREAD_CONSTRUCTOR: GPUJpegBase(_log,parent,"GPUJpeg Encoder"),width(0),height(0),quality(90)
{
	IO_THREAD_INIT("GPUJpegEncoder")

	if (!init())
		throw InitializationFailed("Library Initialization failed");
}

GPUJpegEncoder::~GPUJpegEncoder()
{
}

bool GPUJpegEncoder::step()
{
	shared_ptr<BasicFrame> frame, out_frame;
	if (in[0]) frame=in[0]->pop_frame();
	if (!frame) return true;
	if (!encoder) {
		if (!init_image_params(frame)) return true;
	}
	assert(width==frame->get_width());
	assert(height==frame->get_height());
	yuri::ubyte_t* image_compressed = 0;
	int image_compressed_size = 0;
	if ( gpujpeg_encoder_encode(encoder.get(), reinterpret_cast<unsigned char*>((*frame)[0].data.get()),
			reinterpret_cast<unsigned char**>(&image_compressed), &image_compressed_size) ) {
		log[warning] << "Encoding failed!" << endl;
		return true;
	}
	out_frame=allocate_frame_from_memory(image_compressed,image_compressed_size,false);
	push_video_frame(0,out_frame,YURI_IMAGE_JPEG,width,height);
	return true;
}


bool GPUJpegEncoder::set_param(Parameter &parameter)
{
	if (parameter.name == "quality") {
		quality=parameter.get<yuri::uint_t>();
		if (quality > 100) quality=100;
	} else
		return GPUJpegBase::set_param(parameter);
	return true;
}

bool GPUJpegEncoder::init()
{
	if (!init_device()) return false;

	gpujpeg_set_default_parameters(&encoder_params);
	encoder_params.quality = quality;

	gpujpeg_image_set_default_parameters(&image_params);
	return true;
}
bool GPUJpegEncoder::init_image_params(shared_ptr<BasicFrame> frame)
{
	assert(frame);
	yuri::format_t fmt = frame->get_format();
	image_params.width = frame->get_width();
	image_params.height = frame->get_height();
	image_params.comp_count=3;
	switch (fmt) {
		case YURI_FMT_RGB:
			image_params.color_space=GPUJPEG_RGB;
			image_params.sampling_factor=GPUJPEG_4_4_4;
			break;
		case YURI_FMT_YUV444:
			image_params.color_space=GPUJPEG_YCBCR_ITU_R;
			image_params.sampling_factor=GPUJPEG_4_4_4;
			break;
		case YURI_FMT_YUV422:
			image_params.color_space=GPUJPEG_YCBCR_ITU_R;
			image_params.sampling_factor=GPUJPEG_4_2_2;
			gpujpeg_parameters_chroma_subsampling(&encoder_params);
			break;
		default:
			log[warning] << "Unsupported pixel format" << endl;
			return false;
	}
	encoder.reset(gpujpeg_encoder_create(&encoder_params,&image_params));
	if (!encoder) {
		log[error] << "Failed to create encoder!" << endl;
		return false;
	}
	width = frame->get_width();
	height = frame->get_height();
	log[debug] << "Encoder created " << endl;
	return true;

}
} /* namespace io */
} /* namespace yuri */
