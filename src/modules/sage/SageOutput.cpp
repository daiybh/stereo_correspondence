/*
 * SageOutput.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: neneko
 */

#include "SageOutput.h"
#include "yuri/config/RegisteredClass.h"
namespace yuri {
namespace sage {

REGISTER("sage_output",SageOutput)

IO_THREAD_GENERATOR(SageOutput)

shared_ptr<yuri::config::Parameters> SageOutput::configure()
{
	shared_ptr<yuri::config::Parameters> p = BasicIOThread::configure();

	(*p)["address"]["SAGE address"]=string("127.0.0.1");
	(*p)["width"]["Image width"]=800;
	(*p)["height"]["Image height"]=600;
	return p;
}


SageOutput::SageOutput(yuri::log::Log &log_,yuri::io::pThreadBase parent,yuri::config::Parameters &parameters)
:BasicIOThread(log_,parent,1,0,"SageOutput"),sail_info(0),width(800),height(600),
 fmt(YURI_FMT_YUV422),sage_fmt(PIXFMT_YUV),sage_address("127.0.0.1")
{
	IO_THREAD_INIT("SageOutput")

		log[yuri::log::info] << "Connecting to SAGE @ " << sage_address << "\n";
	sail_info = createSAIL("yuri",width,height,sage_fmt,0,TOP_TO_BOTTOM);//sage_address.c_str());
	if (!sail_info) throw yuri::exception::InitializationFailed("Failed to connect to SAIL");
}


SageOutput::~SageOutput()
{
	deleteSAIL(sail_info);
}
namespace {
struct swap_yuv {
	yuri::ubyte_t a;
	yuri::ubyte_t b;
	swap_yuv(const ushort_t rhs):a((rhs&0xFF00)>>8),b(rhs&0xFF) {}
} __attribute__((packed));
}
bool SageOutput::step()
{
	processMessages(sail_info,0,0);
	if (!in[0]) return true;
	shared_ptr<yuri::io::BasicFrame> frame = in[0]->pop_latest();
	if (!frame) return true;
	if (frame->get_format() != fmt) return true;
	const yuri::FormatInfo_t finfo = yuri::io::BasicPipe::get_format_info(fmt);
	const yuri::size_t sage_line_width = finfo->bpp*width/8;
	const yuri::size_t input_line_width = finfo->bpp*frame->get_width()/8;
	const yuri::size_t copy_width = std::min(sage_line_width, input_line_width);
	const yuri::size_t copy_lines = std::min(height, frame->get_height());
	if (fmt == YURI_FMT_YUV422) {
		// Sage expects UYUV instead od YUYV
		swap_yuv* sail_buffer = reinterpret_cast<swap_yuv *>(nextBuffer(sail_info));
		if (!sail_buffer) {
			log[yuri::log::fatal] << "Got empty buffer from the SAIL library. Assuming connection is closed and bailing out.\n";
			return false;
		}
		for (yuri::size_t line = 0; line < copy_lines; ++line) {
			const yuri::ushort_t* data_start = reinterpret_cast<yuri::ushort_t*>(PLANE_RAW_DATA(frame,0) + line*input_line_width);
			std::copy(data_start,data_start+copy_width/2,sail_buffer+line*sage_line_width/2);
		}
	}
	swapBuffer(sail_info);
	return true;
}

bool SageOutput::set_param(yuri::config::Parameter &parameter)
{
	if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else if (parameter.name == "width")
		width=parameter.get<yuri::size_t>();
	else if (parameter.name == "height")
		height=parameter.get<yuri::size_t>();
	else /*if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else */ return BasicIOThread::set_param(parameter);
	return true;
}

} /* namespace sage */
} /* namespace yuri */

