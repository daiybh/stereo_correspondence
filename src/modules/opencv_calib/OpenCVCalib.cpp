/*!
 * @file 		OpenCVCalib.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "OpenCVCalib.h"
#include "opencv2/calib3d/calib3d.hpp"
#include <boost/assign.hpp>
#include "yuri/core/Module.h"
namespace yuri {
namespace opencv {

REGISTER("opencv_calib",OpenCVCalib)

IO_THREAD_GENERATOR(OpenCVCalib)


core::pParameters OpenCVCalib::configure()
{
	core::pParameters p = core::IOThread::configure();
	p->set_description("Dummy module. For testing only.");
	(*p)["size"]["Set size of ....  (ignored ;)"]=666;
	(*p)["name"]["Set name"]=std::string("");
	p->set_max_pipes(1,1);
	return p;
}


OpenCVCalib::OpenCVCalib(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("OpenCVCalib")),format(YURI_FMT_RGB24)
{
	IO_THREAD_INIT("OpenCVCalib")
}

OpenCVCalib::~OpenCVCalib()
{
}
//
//namespace {
//typedef std::pair<format_t, format_t> fmt_pair;
//std::map<fmt_pair, int > format_map = boost::assign::map_list_of<fmt_pair, int>
//(std::make_pair(YURI_FMT_RGB,YURI_FMT_RGBA),static_cast<int>(CV_BGR2BGRA))
//(std::make_pair(YURI_FMT_RGBA,YURI_FMT_RGB),static_cast<int>(CV_BGRA2BGR));
//typedef std::map<fmt_pair, int >::iterator fmt_map_iter;
//}

bool OpenCVCalib::step()
{
	const core::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {

		const size_t width = frame->get_width();
		const size_t height = frame->get_height();
		const FormatInfo_t in_fi 	= core::BasicPipe::get_format_info(frame->get_format());
		if (!in_fi) {
			log[log::warning] << "Unknown type passed\n";
			return true;
		}
		if (in_fi->compressed) {
			log[log::warning] << "Format marked as compressed, ignoring the frame\n";
			return true;
		}
		if (in_fi->components.size() == 0  || in_fi->component_depths.size() == 0) {
			log[log::warning] << "Wrongly specified format, ignoring\n";
			return true;
		}
		int in_base = -1;
		switch (in_fi->component_depths[0]) {
			case 8: in_base = CV_8UC1;break;
			case 16: in_base = CV_16UC1; break;
		}
		if (in_base < 0) {
			log[log::warning] << "Unsupported bit depth\n";
			return true;
		}
		int in_type 	= CV_MAKETYPE(in_base,  in_fi->components[0].size());

		cv::Mat in_mat(height,width,in_type,PLANE_RAW_DATA(frame,0));
		cv::Size patternsize(4,5); //interior number of corners
		cv::vector<cv::Point2f> corners;
		bool patternfound = cv::findChessboardCorners(in_mat, patternsize, corners,
		        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
		        | cv::CALIB_CB_FAST_CHECK);
		cv::drawChessboardCorners(in_mat,patternsize, corners,patternfound);
		log[log::info] << "pattern " << (patternfound?"found":"not found") << "\n";
		//cv::cvtColor(in_mat,out_mat,it->second);


		push_raw_video_frame(0, frame);

	}
	return true;
}
bool OpenCVCalib::set_param(const core::Parameter& param)
{
	if (param.name =="format") {
		format = core::BasicPipe::get_format_from_string(param.get<std::string>());
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
