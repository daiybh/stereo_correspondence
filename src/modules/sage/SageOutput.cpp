/*!
 * @file 		SageOutput.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.1.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "SageOutput.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"


// Unexported method from libsage, needed to register sharing without using processMessages
void addNewClient(sail *sageInf, char *fsIP);
#include "boost/assign.hpp"

namespace yuri {
namespace sage {

IOTHREAD_GENERATOR(SageOutput)

MODULE_REGISTRATION_BEGIN("sage_output")
		REGISTER_IOTHREAD("sage_output",SageOutput)
MODULE_REGISTRATION_END()


core::Parameters SageOutput::configure()
{
	core::Parameters p = IOThread::configure();

	p["address"]["SAGE address (ignored)"]=std::string("127.0.0.1");
	p["app_name"]["Application name to use when registering to SAGE"]=std::string("yuri");
	p["width"]["Force image width. 0 to use input image size"]=0;
	p["height"]["Force image height. 0 to use input image size"]=0;
	return p;
}

namespace {
std::map<format_t, sagePixFmt> yuri_sage_fmt_map = {
{core::raw_format::uyvy422, PIXFMT_YUV},
{core::raw_format::yuyv422, PIXFMT_YUV},
{core::raw_format::rgb24, PIXFMT_888},
{core::raw_format::bgr24, PIXFMT_888_INV},
//{core::raw_format::rgba32, PIXFMT_8888},
//{core::raw_format::bgra32, PIXFMT_8888_INV},
{core::compressed_frame::dxt1, PIXFMT_DXT}};

}

SageOutput::SageOutput(yuri::log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
:IOFilter(log_,parent,"SageOutput"),sail_info(nullptr),width(0),height(0),
 fmt(0),sage_fmt(PIXFMT_NULL),sage_address("127.0.0.1"),app_name_("yuri")
{
	set_latency(2_ms);
	IOTHREAD_INIT(parameters)
	std::vector<format_t> fmts;
	std::transform(yuri_sage_fmt_map.begin(),
				   yuri_sage_fmt_map.end(),
				   std::back_inserter(fmts),
				   [](const std::pair<format_t, sagePixFmt>& f){return f.first;});

	set_supported_formats(fmts);
}


SageOutput::~SageOutput() noexcept
{
	if (sail_info) deleteSAIL(sail_info);
}

namespace {
struct swap_yuv {
	uint8_t a;
	uint8_t b;
	swap_yuv(const uint16_t rhs):a((rhs&0xFF00)>>8),b(rhs&0xFF) {}
} __attribute__((packed));


bool process_sail_messages(sail* sail_info) {
	sageMessage msg;
	if (sail_info->checkMsg(msg, false) > 0) {
		switch (msg.getCode()) {
			case APP_QUIT:

				return false;
				break;
			case EVT_APP_SHARE:
				addNewClient(sail_info,reinterpret_cast<char*>(msg.getData()));
				break;
		}
	}
	return true;
}

}
bool SageOutput::step()
{
	if (sail_info &&  !process_sail_messages(sail_info)) {
		log[log::info] << "Sail lost connection, quitting";
		request_end();
		return false;
	}
	return IOFilter::step();
}

core::pFrame SageOutput::do_simple_single_step(const core::pFrame& frame)
{
	if (fmt == 0 || !sail_info) {
		core::pVideoFrame video_frame = std::dynamic_pointer_cast<core::VideoFrame>(frame);
		if (!video_frame) return {};
		const format_t tmp_fmt = frame->get_format();
		if (!yuri_sage_fmt_map.count(tmp_fmt)) {
			log[log::warning] << "Unsupported input format";
			return {};
		}
		fmt = tmp_fmt;
		sage_fmt=yuri_sage_fmt_map[fmt];

		if (!width) width = video_frame->get_resolution().width;
		if (!height) height = video_frame->get_resolution().height;
		if (!width || !height) {
			log[log::error] << "Input resolution not specified!";
			return {};
		}
//		log[yuri::log::info] << "Connecting to SAGE @ " << sage_address << "\n";
		log[yuri::log::info] << "Connecting to SAGE with resolution " << width << "x" << height;
		sail_info = createSAIL(app_name_.c_str(),width,height,sage_fmt,0,TOP_TO_BOTTOM);//sage_address.c_str());
		if (!sail_info) {
			//throw yuri::exception::InitializationFailed(
			log[log::fatal] << "Failed to connect to SAIL";
			request_end();
			return {};
		}
	}

	if (frame->get_format() != fmt) return {};

	if (core::pRawVideoFrame raw_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(frame)) {
		const auto& finfo = core::raw_format::get_format_info(fmt);
//		const yuri::FormatInfo_t finfo = core::BasicPipe::get_format_info(fmt);
		auto depth = finfo.planes[0].bit_depth;
		const yuri::size_t sage_line_width = depth.first * width/depth.second/8;
		const yuri::size_t input_line_width = depth.first * raw_frame->get_width()/depth.second/8;;
		const yuri::size_t copy_width = std::min(sage_line_width, input_line_width);
		const yuri::size_t copy_lines = std::min(height, raw_frame->get_height());
		if (fmt == core::raw_format::yuyv422) {
			// Sage expects UYUV instead od YUYV
			swap_yuv* sail_buffer = reinterpret_cast<swap_yuv *>(nextBuffer(sail_info));
			if (!sail_buffer) {
				log[yuri::log::fatal] << "Got empty buffer from the SAIL library. Assuming connection is closed and bailing out.\n";
				request_end();
				return {};
			}
			for (yuri::size_t line = 0; line < copy_lines; ++line) {
				const uint16_t* data_start = reinterpret_cast<uint16_t*>(PLANE_RAW_DATA(raw_frame,0) + line*input_line_width);
				std::copy(data_start,data_start+copy_width/2,sail_buffer+line*sage_line_width/2);
			}
		}
		else  {
			uint8_t* sail_buffer = reinterpret_cast<uint8_t*>(nextBuffer(sail_info));
			if (!sail_buffer) {
				log[yuri::log::fatal] << "Got empty buffer from the SAIL library. Assuming connection is closed and bailing out.\n";
				request_end();
				return {};
			}
//			log[log::info] << "input_line_width: " << input_line_width << ", sage_line_width: " << sage_line_width << " copy_lines: " <<copy_lines;
			for (yuri::size_t line = 0; line < copy_lines; ++line) {
				const uint8_t* data_start = PLANE_RAW_DATA(raw_frame,0) + line*input_line_width;
				std::copy(data_start,data_start+copy_width,sail_buffer+line*sage_line_width);
			}
		}
	} else {
		log[log::warning] << "Compressed frames not supported yet";
	}

//		} else {
//			ubyte_t* sail_buffer = reinterpret_cast<ubyte_t*>(nextBuffer(sail_info));
//			const yuri::ubyte_t* data_start = reinterpret_cast<yuri::ubyte_t*>(PLANE_RAW_DATA(frame,0));
//			std::copy(data_start,data_start+PLANE_SIZE(frame,0),sail_buffer);
//		}
	//swapBuffer(sail_info);
	sail_info->swapBuffer(SAGE_NON_BLOCKING);
	return {};
}

bool SageOutput::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name() == "address")
		sage_address=parameter.get<std::string>();
	else if (parameter.get_name() == "app_name")
		app_name_ =parameter.get<std::string>();
	else if (parameter.get_name() == "width")
		width=parameter.get<yuri::size_t>();
	else if (parameter.get_name() == "height")
		height=parameter.get<yuri::size_t>();
	else /*if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else */ return IOThread::set_param(parameter);
	return true;
}

} /* namespace sage */
} /* namespace yuri */

