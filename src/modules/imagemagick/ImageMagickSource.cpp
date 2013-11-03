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
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "Magick++.h"
#include <map>
namespace yuri {
namespace imagemagick_module {


MODULE_REGISTRATION_BEGIN("imagemagick_source")
		REGISTER_IOTHREAD("imagemagick_source",ImageMagickSource)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(ImageMagickSource)

namespace {
std::map<format_t, std::pair<std::string, Magick::StorageType> > yuri_to_magick_format = {
{core::raw_format::rgb24,	{"RGB",	Magick::CharPixel}},
{core::raw_format::rgba32,	{"RGBA",Magick::CharPixel}},
{core::raw_format::bgr24,	{"BGR",	Magick::CharPixel}}
};
}

core::Parameters ImageMagickSource::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::CompressedVideoFrame>::configure();
	p.set_description("Image loader based on ImageMagick.");
	p["format"]["Set output format"]="RGB";
	return p;
}


ImageMagickSource::ImageMagickSource(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::CompressedVideoFrame>(log_,parent,std::string("ImageMagickSource")),
format_(core::raw_format::rgb24)
{
	IOTHREAD_INIT(parameters)
}

ImageMagickSource::~ImageMagickSource() noexcept
{
}

core::pFrame ImageMagickSource::do_special_single_step(const core::pCompressedVideoFrame& frame)
{
	try {
		Magick::Blob blob(frame->data(),frame->size());
		Magick::Image image(blob);
		if (!still_running()) return {};
		image.modifyImage();
		resolution_t resolution = { image.columns(), image.rows()};
		log[log::debug] << "Loaded image " << resolution.width << "x" << resolution.height;
		if (!still_running()) return {};
		core::pRawVideoFrame out_frame = core::RawVideoFrame::create_empty(format_,resolution);
		auto img_type = yuri_to_magick_format[format_];
		image.write(0,0,resolution.width,resolution.height,img_type.first,img_type.second,PLANE_RAW_DATA(out_frame,0));
		return {out_frame};
	}
	catch (std::exception& e) {
		log[log::error] << "Exception during decoding: " << e.what();
	}
	return {};
}
bool ImageMagickSource::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
		if (yuri_to_magick_format.find(format_)==yuri_to_magick_format.end()) {
			log[log::warning] << "Specified format is not currently supported. Falling back to RGB";
			format_ = core::raw_format::rgb24;
		}
	} else return core::SpecializedIOFilter<core::CompressedVideoFrame>::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
