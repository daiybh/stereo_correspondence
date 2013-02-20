/*!
 * @file 		OpenCV.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "OpenCV.h"
#include "yuri/config/RegisteredClass.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/assign.hpp>

namespace yuri {
namespace dummy_module {

REGISTER("opencv",OpenCV)

IO_THREAD_GENERATOR(OpenCV)

using namespace yuri::log;

shared_ptr<config::Parameters> OpenCV::configure()
{
	shared_ptr<config::Parameters> p = io::BasicIOThread::configure();
	p->set_description("Dummy module. For testing only.");
	(*p)["size"]["Set size of ....  (ignored ;)"]=666;
	(*p)["name"]["Set name"]=std::string("");
	p->set_max_pipes(1,1);
	return p;
}


OpenCV::OpenCV(log::Log &log_,io::pThreadBase parent,config::Parameters &parameters):
io::BasicIOThread(log_,parent,1,1,std::string("opencv")),format(YURI_FMT_RGB24)
{
	IO_THREAD_INIT("OpenCV")
}

OpenCV::~OpenCV()
{
}

namespace {
typedef std::pair<format_t, format_t> fmt_pair;
std::map<fmt_pair, int > format_map = boost::assign::map_list_of<fmt_pair, int>
(std::make_pair(YURI_FMT_RGB,YURI_FMT_RGBA),CV_BGR2BGRA)
(std::make_pair(YURI_FMT_RGBA,YURI_FMT_RGB),CV_BGRA2BGR);
typedef std::map<fmt_pair, int >::iterator fmt_map_iter;
}

bool OpenCV::step()
{
	const io::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		fmt_pair fmts = std::make_pair(frame->get_format(), format);
		fmt_map_iter it = format_map.find(fmts);
		if (it==format_map.end()) {
			log[warning] << "Unsupported conversion combination! (" <<
					io::BasicPipe::get_format_string(fmts.first) << " -> " <<
					io::BasicPipe::get_format_string(fmts.second) << "\n";
			return true;
		}
		const size_t width = frame->get_width();
		const size_t height = frame->get_height();
		const FormatInfo_t in_fi 	= io::BasicPipe::get_format_info(frame->get_format());
		const FormatInfo_t out_fi	= io::BasicPipe::get_format_info(format);
		if (!in_fi || !out_fi) {
			log[warning] << "Unknown type passed\n";
			return true;
		}
		if (in_fi->compressed || out_fi->compressed) {
			log[warning] << "Format marked as compressed, ignoring the frame\n";
			return true;
		}
		if (in_fi->components.size() == 0 || out_fi->components.size() == 0
				|| in_fi->component_depths.size() == 0
				|| out_fi->component_depths.size() == 0) {
			log[warning] << "Wrongly specified format, ignoring\n";
			return true;
		}
		int in_base = -1;
		switch (in_fi->component_depths[0]) {
			case 8: in_base = CV_8UC1;break;
			case 16: in_base = CV_16UC1; break;
		}
		int out_base = -1;
		switch (out_fi->component_depths[0]) {
			case 8: out_base = CV_8UC1;break;
			case 16: out_base = CV_16UC1; break;
		}
		if (in_base < 0 || out_base < 0) {
			log[warning] << "Unsupported bit depth\n";
			return true;
		}
		int in_type 	= CV_MAKETYPE(in_base,  in_fi->components[0].size());
		int out_type	= CV_MAKETYPE(out_base, out_fi->components[0].size());
		io::pBasicFrame output = allocate_empty_frame(format,width,height);
		cv::Mat in_mat(height,width,in_type,PLANE_RAW_DATA(frame,0));
		cv::Mat out_mat(height,width,out_type,PLANE_RAW_DATA(output,0));
		cv::cvtColor(in_mat,out_mat,it->second);
		push_raw_video_frame(0, output);

	}
	return true;
}
bool OpenCV::set_param(config::Parameter& param)
{
	if (param.name =="format") {
		format = io::BasicPipe::get_format_from_string(param.get<std::string>());
	} else return io::BasicIOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
