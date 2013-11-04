/*!
 * @file 		OpenCVFaceDetect.cpp
 * @author 		<Your name>
 * @date		04.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "OpenCVFaceDetect.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri {
namespace opencv {


IOTHREAD_GENERATOR(OpenCVFaceDetect)

core::Parameters OpenCVFaceDetect::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("OpenCVFaceDetect");
	p["haar_cascade"]["Path to he xml file with HAAR cascade"]="haarcascade_frontalface_default.xml";
	return p;
}


OpenCVFaceDetect::OpenCVFaceDetect(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("opencv_facedetection")),
event::BasicEventProducer(log),
haar_cascade_file_("haarcascade_frontalface_default.xml")
{
	IOTHREAD_INIT(parameters)
	haar_cascade_.load(haar_cascade_file_);
	set_supported_formats({core::raw_format::y8});
}

OpenCVFaceDetect::~OpenCVFaceDetect() noexcept
{
}

core::pFrame OpenCVFaceDetect::do_special_single_step(const core::pRawVideoFrame& frame)
{
	resolution_t res = frame->get_resolution();
	cv::Mat in_mat(res.height,res.width,CV_8UC1,PLANE_RAW_DATA(frame,0));
	std::vector< cv::Rect> faces;
	haar_cascade_.detectMultiScale(in_mat, faces);
	if (faces.empty()) {
		log[log::warning] << "No faces found!";
	} else {
		log[log::info] << "Found " << faces.size() << " faces";
		for (auto x: faces) {
			log[log::info] << x.width << "x" << x.height << "+" << x.x << "+" << x.y;
		}
		emit_event("x", make_shared<event::EventInt>(faces[0].x+faces[0].width/2));
		emit_event("y", make_shared<event::EventInt>(faces[0].y+faces[0].height/2));
		emit_event("width", make_shared<event::EventInt>(faces[0].width));
		emit_event("height", make_shared<event::EventInt>(faces[0].height));
	}

	return {};

}
bool OpenCVFaceDetect::set_param(const core::Parameter& param)
{
	if (param.get_name() == "haar_cascade") {
		haar_cascade_file_ = param.get<std::string>();
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
	return true;
}

} /* namespace opencv_facedetection */
} /* namespace yuri */

