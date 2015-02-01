/*
 * GPUJpegDecoder.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: neneko
 */

#include "GPUJpegDecoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"

namespace yuri {
namespace gpujpeg {

//REGISTER("gpujpeg_decoder",GPUJpegDecoder)
IOTHREAD_GENERATOR(GPUJpegDecoder)

core::Parameters GPUJpegDecoder::configure()
{
	core::Parameters p = GPUJpegBase::configure();
//	p->set_max_pipes(1,1);
//	p->add_output_format(YURI_FMT_YUV422);
//	p->add_input_format(YURI_IMAGE_JPEG);
	p["format"]["Output format (yuv422, yuv444, rgb)"]="yuv422";
	return p;
}

GPUJpegDecoder::GPUJpegDecoder(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
:GPUJpegBase(log_,parent,"GPUJpeg Decoder")
{
	IOTHREAD_INIT(parameters)

	if (!init())
		throw exception::InitializationFailed("Library Initialization failed");
	if (!out_format)
		throw exception::InitializationFailed("Wrong output format");
}

GPUJpegDecoder::~GPUJpegDecoder() noexcept
{
}

bool GPUJpegDecoder::init()
{
	switch (out_format) {
		case core::raw_format::rgb24:
			color_space = GPUJPEG_RGB;
			sampling = GPUJPEG_4_4_4;
			break;
		case core::raw_format::yuv444:
			color_space = GPUJPEG_YCBCR;
			sampling = GPUJPEG_4_4_4;
			break;
		case core::raw_format::yuyv422:
			color_space = GPUJPEG_YCBCR;
			sampling = GPUJPEG_4_2_2;
			break;
		default:
			out_format = 0;
			break;
	}
	if (!out_format) return false;
	if (!init_device()) return false;

	decoder.reset(gpujpeg_decoder_create(),[](gpujpeg_decoder* p){gpujpeg_decoder_destroy(p);});
	if (!decoder) {
		log[log::error] << "Failed to create decoder";
		throw exception::InitializationFailed("Failed to create decoder");
	}
	gpujpeg_decoder_output_set_default(&decoder_output);
	gpujpeg_decoder_set_output_format(decoder.get(), color_space, sampling);

	return true;
}
bool GPUJpegDecoder::do_initialize_converter(format_t target_format)
{
	out_format = target_format;
	return init();
}
core::pFrame GPUJpegDecoder::do_simple_single_step(core::pFrame frame)
{
	return do_convert_frame(std::move(frame), out_format);
}
core::pFrame GPUJpegDecoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	if (target_format != out_format) return {};
	auto frame = std::dynamic_pointer_cast<core::CompressedVideoFrame>(input_frame);
	if (!frame) return {};
	if (frame->get_format() != core::compressed_frame::jpeg) {
		log[log::warning] << "Received frame in wrong format.";
		return {};
	}
	if (gpujpeg_decoder_decode(decoder.get(), frame->get_data().data(), frame->size(),
		   &decoder_output)) {
		log[log::warning] << "Decoding failed";
	}
	//log[info] << "Got frame " << decoder->coder.data_width << "x" << decoder->coder.data_height << endl;
	auto out_frame= core::RawVideoFrame::create_empty(target_format, frame->get_resolution(),decoder_output.data,
			decoder_output.data_size);
//	push_video_frame(0,out_frame,out_format,decoder->coder.data_width, decoder->coder.data_height);

	return out_frame;
}

bool GPUJpegDecoder::set_param(const core::Parameter& parameter)
{
	if (parameter.get_name() == "format") {
		out_format = core::raw_format::parse_format(parameter.get<std::string>());
//		if (out_format) {

//		}
	} else
		return GPUJpegBase::set_param(parameter);
	return true;
}

} /* namespace io */
} /* namespace yuri */
