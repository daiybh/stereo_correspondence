/*!
 * @file 		OpenCV.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "OpenCVConvert.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
namespace yuri {
namespace opencv {

IOTHREAD_GENERATOR(OpenCVConvert)


core::Parameters OpenCVConvert::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("Opencv color conversion module.");
	p["format"]["Output format"]=std::string("RGB");
	return p;
}


OpenCVConvert::OpenCVConvert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("opencv_convert")),
format_(core::raw_format::rgb24)
{
	IOTHREAD_INIT(parameters)
}

OpenCVConvert::~OpenCVConvert() noexcept
{
}

core::pFrame OpenCVConvert::do_special_single_step(const core::pRawVideoFrame& frame)
{
	assert (frame);
	if (frame->get_format() == format_) {
		return frame;
	}
	fmt_pair fmts = std::make_pair(frame->get_format(), format_);
	auto it = convert_format_map.find(fmts);
	if (it==convert_format_map.end()) {
		log[log::warning] << "Unsupported conversion combination! (" <<
				core::raw_format::get_format_name(fmts.first) << " -> " <<
				core::raw_format::get_format_name(fmts.second) << "\n";
		return {};
	}
	const size_t width = frame->get_width();
	const size_t height = frame->get_height();
	const auto& in_fi 	= core::raw_format::get_format_info(frame->get_format());
	const auto& out_fi	= core::raw_format::get_format_info(format_);

	if (in_fi.planes[0].components.size() == 0 || out_fi.planes[0].components.size() == 0
			|| in_fi.planes[0].component_bit_depths.size() == 0
			|| out_fi.planes[0].component_bit_depths.size() == 0) {
		log[log::warning] << "Wrongly specified format, ignoring\n";
		return {};
	}
	int in_base = -1;
	switch (in_fi.planes[0].component_bit_depths.size()) {
		case 8: in_base = CV_8UC1;break;
		case 16: in_base = CV_16UC1; break;
	}
	int out_base = -1;
	switch (out_fi.planes[0].component_bit_depths.size()) {
		case 8: out_base = CV_8UC1;break;
		case 16: out_base = CV_16UC1; break;
	}
	if (in_base < 0 || out_base < 0) {
		log[log::warning] << "Unsupported bit depth\n";
		return {};
	}
	int in_type 	= CV_MAKETYPE(in_base,  in_fi.planes[0].components.size());
	int out_type	= CV_MAKETYPE(out_base, out_fi.planes[0].components.size());
	core::pRawVideoFrame output = core::RawVideoFrame::create_empty(format_,{width,height},true);
	cv::Mat in_mat(height,width,in_type,PLANE_RAW_DATA(frame,0));
	cv::Mat out_mat(height,width,out_type,PLANE_RAW_DATA(output,0));
	cv::cvtColor(in_mat,out_mat,it->second);
	return output;
}
core::pFrame OpenCVConvert::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	if (!input_frame) return {};
	auto it = convert_format_map.find({input_frame->get_format(), target_format});
	if (it != convert_format_map.end()) return {};
	format_ = target_format;
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame> (input_frame);
	if (frame) return do_special_single_step(frame);
	return {};

}
bool OpenCVConvert::set_param(const core::Parameter& param)
{
	if (param.get_name() =="format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
