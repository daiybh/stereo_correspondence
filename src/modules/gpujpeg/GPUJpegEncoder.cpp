/*
 * GPUJpegEncoder.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#include "GPUJpegEncoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include <cassert>
namespace yuri {
namespace gpujpeg {


IOTHREAD_GENERATOR(GPUJpegEncoder)

core::Parameters GPUJpegEncoder::configure()
{
	core::Parameters p = GPUJpegBase::configure();
	p["quality"]["Jpeg quality (0-100)"]=90;
//	p->set_max_pipes(1,1);
//	p->add_input_format(YURI_FMT_YUV422);
//	p->add_output_format(YURI_IMAGE_JPEG);
	return p;
}


GPUJpegEncoder::GPUJpegEncoder(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
:GPUJpegBase(log_,parent,"GPUJpeg Encoder"),width(0),height(0),quality(90)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::raw_format::rgb24, core::raw_format::yuv444, core::raw_format::uyvy422});
	if (!init())
		throw exception::InitializationFailed("Library Initialization failed");
}

GPUJpegEncoder::~GPUJpegEncoder() noexcept
{
}

core::pFrame GPUJpegEncoder::do_simple_single_step(core::pFrame frame)
{
	return do_convert_frame(std::move(frame), core::compressed_frame::jpeg);
}
core::pFrame GPUJpegEncoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{

	auto frame = std::dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) return {};
	if (!encoder) {
		if (!init_image_params(frame)) return {};
	}
	assert(width==frame->get_width());
	assert(height==frame->get_height());
	uint8_t* image_compressed = nullptr;
	int image_compressed_size = 0;
	gpujpeg_encoder_input input = {
			GPUJPEG_ENCODER_INPUT_IMAGE,
			reinterpret_cast<unsigned char*>(PLANE_RAW_DATA(frame,0)),
			0
	};
	if ( gpujpeg_encoder_encode(encoder.get(), &input,
			reinterpret_cast<unsigned char**>(&image_compressed), &image_compressed_size) ) {
		log[log::warning] << "Encoding failed!";
		return {};
	}
	auto out_frame=core::CompressedVideoFrame::create_empty(core::compressed_frame::jpeg, frame->get_resolution(), image_compressed, image_compressed_size);
	//push_video_frame(0,out_frame,YURI_IMAGE_JPEG,width,height);
	return out_frame;
}
bool GPUJpegEncoder::do_initialize_converter(format_t target_format)
{
	return true;
}

bool GPUJpegEncoder::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name() == "quality") {
		quality=parameter.get<uint16_t>();
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
bool GPUJpegEncoder::init_image_params(const core::pRawVideoFrame& frame)
{
	assert(frame);
	yuri::format_t fmt = frame->get_format();
	image_params.width = frame->get_width();
	image_params.height = frame->get_height();
	image_params.comp_count=3;
	switch (fmt) {
		case core::raw_format::rgb24:
			image_params.color_space=GPUJPEG_RGB;
			image_params.sampling_factor=GPUJPEG_4_4_4;
			break;
		case core::raw_format::yuv444:
			image_params.color_space=GPUJPEG_YCBCR;
			image_params.sampling_factor=GPUJPEG_4_4_4;
			break;
		case core::raw_format::uyvy422:
			image_params.color_space=GPUJPEG_YCBCR;
			image_params.sampling_factor=GPUJPEG_4_2_2;
			gpujpeg_parameters_chroma_subsampling(&encoder_params);
			break;
		default:
			log[log::warning] << "Unsupported pixel format";
			return false;
	}
	encoder.reset(gpujpeg_encoder_create(&encoder_params,&image_params), [](gpujpeg_encoder*p){gpujpeg_encoder_destroy(p);});
	if (!encoder) {
		log[log::error] << "Failed to create encoder!";
		return false;
	}
	width = frame->get_width();
	height = frame->get_height();
	log[log::debug] << "Encoder created ";
	return true;

}
} /* namespace io */
} /* namespace yuri */
