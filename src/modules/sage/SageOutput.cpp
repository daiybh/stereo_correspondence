/*!
 * @file 		SageOutput.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.1.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "SageOutput.h"
#include "yuri/core/Module.h"

// Unexported method from libsage, needed to register sharing without using processMessages
void addNewClient(sail *sageInf, char *fsIP);
#include "boost/assign.hpp"

namespace yuri {
namespace sage {

REGISTER("sage_output",SageOutput)

IO_THREAD_GENERATOR(SageOutput)

core::pParameters SageOutput::configure()
{
	core::pParameters p = BasicIOThread::configure();

	(*p)["address"]["SAGE address (ignored)"]=std::string("127.0.0.1");
	(*p)["app_name"]["Application name to use when registering to SAGE"]=std::string("yuri");
	(*p)["width"]["Force image width. -1 to use input image size"]=-1;
	(*p)["height"]["Force image height. -1 to use input image size"]=-1;
	return p;
}

namespace {
std::map<format_t, sagePixFmt> yuri_sage_fmt_map = boost::assign::map_list_of<format_t, sagePixFmt>
(YURI_FMT_UYVY422, PIXFMT_YUV)
(YURI_FMT_YUV422, PIXFMT_YUV)
(YURI_FMT_RGB24, PIXFMT_888)
(YURI_FMT_BGR, PIXFMT_888_INV)
(YURI_FMT_RGB32, PIXFMT_8888)
(YURI_FMT_BGRA, PIXFMT_8888_INV)
(YURI_FMT_DXT1, PIXFMT_DXT);
}

SageOutput::SageOutput(yuri::log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters)
:BasicIOThread(log_,parent,1,0,"SageOutput"),sail_info(0),width(-1),height(-1),
 fmt(YURI_FMT_NONE),sage_fmt(PIXFMT_NULL),sage_address("127.0.0.1"),app_name_("yuri")
{
	IO_THREAD_INIT("SageOutput")
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
	if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_latest();
	if (!frame) return true;
	if (fmt == YURI_FMT_NONE) {
		const format_t tmp_fmt = frame->get_format();
		if (!yuri_sage_fmt_map.count(tmp_fmt)) {
			log[log::warning] << "Unsupported input format";
			return true;
		}
		fmt = tmp_fmt;
		sage_fmt=yuri_sage_fmt_map[fmt];
		log[yuri::log::info] << "Connecting to SAGE @ " << sage_address << "\n";
		sail_info = createSAIL(app_name_.c_str(),width,height,sage_fmt,0,TOP_TO_BOTTOM);//sage_address.c_str());
		if (!sail_info) {
			//throw yuri::exception::InitializationFailed(
			log[log::fatal] << "Failed to connect to SAIL";
			request_end();
			return false;
		}
	}
	if (frame->get_format() != fmt) return true;
	const yuri::FormatInfo_t finfo = core::BasicPipe::get_format_info(fmt);
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
	else if (!finfo->compressed){
		ubyte_t* sail_buffer = reinterpret_cast<ubyte_t*>(nextBuffer(sail_info));
		if (!sail_buffer) {
			log[yuri::log::fatal] << "Got empty buffer from the SAIL library. Assuming connection is closed and bailing out.\n";
			return false;
		}
		for (yuri::size_t line = 0; line < copy_lines; ++line) {
			const yuri::ubyte_t* data_start = reinterpret_cast<yuri::ubyte_t*>(PLANE_RAW_DATA(frame,0) + line*input_line_width);
			std::copy(data_start,data_start+copy_width,sail_buffer+line*sage_line_width);
		}
	} else {
		ubyte_t* sail_buffer = reinterpret_cast<ubyte_t*>(nextBuffer(sail_info));
		const yuri::ubyte_t* data_start = reinterpret_cast<yuri::ubyte_t*>(PLANE_RAW_DATA(frame,0));
		std::copy(data_start,data_start+PLANE_SIZE(frame,0),sail_buffer);
	}
	//swapBuffer(sail_info);
	sail_info->swapBuffer(SAGE_NON_BLOCKING);
	return true;
}

bool SageOutput::set_param(const core::Parameter &parameter)
{
	if (parameter.name == "address")
		sage_address=parameter.get<std::string>();
	else if (parameter.name == "app_name")
		app_name_ =parameter.get<std::string>();
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

