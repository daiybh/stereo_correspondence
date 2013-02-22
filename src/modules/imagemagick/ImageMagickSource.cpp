/*!
 * @file 		DummyModule.cpp
 * @author 		Zdenek Travnicek
 * @date		17.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "ImageMagickSource.h"
#include "yuri/core/Module.h"
#include "Magick++.h"
#include <map>
#include <boost/assign.hpp>
namespace yuri {
namespace imagemagick_module {

REGISTER("imagemagick_source",ImageMagickSource)

IO_THREAD_GENERATOR(ImageMagickSource)


namespace {
std::map<yuri::format_t, std::pair<std::string, Magick::StorageType> > yuri_to_magick_format = boost::assign::map_list_of<yuri::format_t, std::pair<std::string, Magick::StorageType> >
(YURI_FMT_RGB24,std::make_pair("RGB",Magick::CharPixel))
(YURI_FMT_RGB32,std::make_pair("RGBA",Magick::CharPixel))
(YURI_FMT_BGR,std::make_pair("BGR",Magick::CharPixel));
}

core::pParameters ImageMagickSource::configure()
{
	core::pParameters p = BasicIOThread::configure();
	p->set_description("Image loader based on ImageMagick.");
	(*p)["format"]["Set output format"]="RGB";
	p->set_max_pipes(1,1);
	return p;
}


ImageMagickSource::ImageMagickSource(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("ImageMagickSource"))
{
	IO_THREAD_INIT("ImageMagickSource")
}

ImageMagickSource::~ImageMagickSource()
{
}

bool ImageMagickSource::step()
{
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;
	yuri::format_t fmt = frame->get_format();
	if (core::BasicPipe::get_format_group(fmt)!=YURI_TYPE_VIDEO) {
		return true;
	}
	if (frame->get_planes_count()>1) {
		log[log::warning] << "Input frame has more than 1 plane, ignoring them\n";
	}
	try {
		Magick::Blob blob(PLANE_RAW_DATA(frame,0),PLANE_SIZE(frame,0));
		Magick::Image image(blob);
		if (!still_running()) return false;
		image.modifyImage();
		yuri::size_t width = image.columns();
		yuri::size_t height = image.rows();
		log[log::debug] << "Loaded image " << width << "x" <<height <<"\n";
		if (!still_running()) return false;
		core::pBasicFrame out_frame = allocate_empty_frame(format,width,height);
		std::pair<std::string, Magick::StorageType> img_type = yuri_to_magick_format[format];
		image.write(0,0,width,height,img_type.first,img_type.second,PLANE_RAW_DATA(out_frame,0));
		push_raw_video_frame(0,out_frame);
	}
	catch (std::exception& e) {
		log[log::error] << "Exception during decoding: " << e.what() << "\n";
	}
	return true;
}
bool ImageMagickSource::set_param(const core::Parameter& param)
{
	if (param.name == "format") {
		format = core::BasicPipe::get_format_from_string(param.get<std::string>());
		if (yuri_to_magick_format.find(format)==yuri_to_magick_format.end()) {
			log[log::warning] << "Specified format is not currently supported. Falling back to RGB\n";
			format = YURI_FMT_RGB24;
		}
	} else return BasicIOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
