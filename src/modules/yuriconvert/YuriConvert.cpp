/*!
 * @file 		YuriConvertor.cpp
 * @author 		Zdenek Travnicek
 * @date 		13.8.2010
 * @date		16.2.2013
 * * @date		26.5.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "YuriConvert.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/thread/ConverterRegister.h"
#include <cassert>
namespace yuri {

namespace video {

IOTHREAD_GENERATOR(YuriConvertor)

MODULE_REGISTRATION_BEGIN("yuri_convert")
		REGISTER_IOTHREAD("yuri_convert",YuriConvertor)
		REGISTER_CONVERTER(core::raw_format::rgb24, 	core::raw_format::rgba32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::bgra32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::rgb24, 	core::raw_format::argb32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::abgr32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::rgb24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::bgr24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::rgb24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::bgr24, "yuri_convert", 12)


		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::argb32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::abgr32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::bgra32, "yuri_convert", 10)

		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::rgba32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::abgr32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::bgra32, "yuri_convert", 10)

		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::rgba32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::argb32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::abgr32, "yuri_convert", 10)

		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::argb32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::rgba32, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::bgra32, "yuri_convert", 10)

		REGISTER_CONVERTER(core::raw_format::rgb24,  	core::raw_format::bgra32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::rgba32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::rgb24,  	core::raw_format::abgr32, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::argb32, "yuri_convert", 12)

		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::bgr24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::rgb24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::bgr24, "yuri_convert", 12)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::rgb24, "yuri_convert", 12)

		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::rgb24, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::rgb24,  	core::raw_format::bgr24, "yuri_convert", 10)


		REGISTER_CONVERTER(core::raw_format::yuyv422,  	core::raw_format::uyvy422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::uyvy422,  	core::raw_format::yuyv422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yvyu422, 	core::raw_format::vyuy422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::vyuy422, 	core::raw_format::yvyu422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::uyvy422, 	core::raw_format::vyuy422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::vyuy422,  	core::raw_format::uyvy422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yuyv422,  	core::raw_format::yvyu422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yvyu422, 	core::raw_format::yuyv422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::uyvy422, 	core::raw_format::yvyu422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::vyuy422,  	core::raw_format::yuyv422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yuyv422, 	core::raw_format::vyuy422, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yvyu422, 	core::raw_format::uyvy422, "yuri_convert", 10)

		REGISTER_CONVERTER(core::raw_format::yuyv422,  	core::raw_format::yuv444, "yuri_convert", 15)
		REGISTER_CONVERTER(core::raw_format::yuv444,  	core::raw_format::yuyv422, "yuri_convert", 15)
		REGISTER_CONVERTER(core::raw_format::uyvy422,  	core::raw_format::yuv444, "yuri_convert", 15)
		REGISTER_CONVERTER(core::raw_format::yuv444,  	core::raw_format::uyvy422, "yuri_convert", 15)

		REGISTER_CONVERTER(core::raw_format::yuva4444,  core::raw_format::yuv444, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::ayuv4444,  core::raw_format::yuv444, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yuv444,  	core::raw_format::yuva4444, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yuv444,  	core::raw_format::ayuv4444, "yuri_convert", 10)
		REGISTER_CONVERTER(core::raw_format::yuva4444,  core::raw_format::yuyv422, "yuri_convert", 15)
		REGISTER_CONVERTER(core::raw_format::ayuv4444,  core::raw_format::yuyv422, "yuri_convert", 15)

		REGISTER_CONVERTER(core::raw_format::rgb24,  	core::raw_format::yuv444, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::yuv444, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::yuv444, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::yuv444, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::yuv444, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::yuv444, "yuri_convert", 20)

		REGISTER_CONVERTER(core::raw_format::rgb24,  	core::raw_format::yuyv422, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::rgba32,  	core::raw_format::yuyv422, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::argb32,  	core::raw_format::yuyv422, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::bgr24,  	core::raw_format::yuyv422, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::bgra32,  	core::raw_format::yuyv422, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::abgr32,  	core::raw_format::yuyv422, "yuri_convert", 25)

		REGISTER_CONVERTER(core::raw_format::yuv444, 	core::raw_format::rgb24, "yuri_convert", 20)
		REGISTER_CONVERTER(core::raw_format::yuyv422, 	core::raw_format::rgb24, "yuri_convert", 25)
		REGISTER_CONVERTER(core::raw_format::uyvy422, 	core::raw_format::rgb24, "yuri_convert", 25)

//		REGISTER_CONVERTER(core::raw_format::yuv422_v210, core::raw_format::yuyv422, "yuri_convert", 30)

		REGISTER_CONVERTER(core::raw_format::u8, 	core::raw_format::y8, "yuri_convert", 1)
		REGISTER_CONVERTER(core::raw_format::v8, 	core::raw_format::y8, "yuri_convert", 1)
		REGISTER_CONVERTER(core::raw_format::r8, 	core::raw_format::y8, "yuri_convert", 1)
		REGISTER_CONVERTER(core::raw_format::g8, 	core::raw_format::y8, "yuri_convert", 1)
		REGISTER_CONVERTER(core::raw_format::b8, 	core::raw_format::y8, "yuri_convert", 1)
		REGISTER_CONVERTER(core::raw_format::depth8, 	core::raw_format::y8, "yuri_convert", 1)




MODULE_REGISTRATION_END()


/* ***************************************************************************
  Adding a new converter:

  For most cases, it should be enough to specialize function template convert_line
  and add ADD_CONVERSION to converters below

  in special cases you may need to specialize get_linesize or allocate_frame
/ *************************************************************************** */



namespace {
	// Values of parameters as size_t as we can;t use double as template parameter. Tha value should represent the float value multiplied by 10000
	const size_t Wr_601 	= 2990;
	const size_t Wb_601 	= 1140;
	const size_t Wr_709 	= 2126;
	const size_t Wb_709 	=  722;
	const size_t Wr_2020 	= 2627;
	const size_t Wb_2020 	=  593;


	typedef function<core::pRawVideoFrame(const core::pRawVideoFrame&, const YuriConvertor&)> converter_t;
	typedef std::pair<yuri::format_t, yuri::format_t> format_pair_t;

//	inline unsigned int convY(unsigned int Y) { return (Y*219+4128) >> 6; }
//	inline unsigned int convC(unsigned int C) {	return (C*7+129) >> 1; }

	template<format_t fmt>
		core::pRawVideoFrame allocate_frame(size_t width, size_t height);
//	template<format_t fmt>
//		size_t get_linesize(size_t width);
	template<format_t fmt_in, format_t fmt_out>
		void convert_line(core::Plane::const_iterator src, core::Plane::iterator dest, size_t width, const YuriConvertor&);
	template<format_t fmt_in, format_t fmt_out>
		void convert_line(core::Plane::const_iterator src, core::Plane::iterator dest, size_t width);
	template<format_t fmt_in, format_t fmt_out>
		core::pRawVideoFrame convert_formats( const core::pRawVideoFrame& frame, const YuriConvertor&);

	#include "YuriConvert.impl"

#define ADD_CONVERSION(fmt1, fmt2) {std::make_pair(fmt1, fmt2), &convert_formats<fmt1, fmt2>},

	std::map<format_pair_t, converter_t> converters = {

		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::rgba32)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::bgra32)
		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::argb32)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::abgr32)
		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::bgr24)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::bgr24)

		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::rgba32)
		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::bgra32)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::argb32)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::abgr32)

		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::rgba32)
		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::abgr32)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::argb32)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::bgra32)

		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::argb32)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::abgr32)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::bgra32)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::rgba32)

		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::bgra32)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::rgba32)
		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::abgr32)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::argb32)

		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::bgr24)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::bgr24)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::rgb24)


		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::bgr24)

		ADD_CONVERSION(core::raw_format::yuyv422, 		core::raw_format::uyvy422)
		ADD_CONVERSION(core::raw_format::uyvy422, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::yvyu422,		core::raw_format::vyuy422)
		ADD_CONVERSION(core::raw_format::vyuy422,		core::raw_format::yvyu422)
		ADD_CONVERSION(core::raw_format::uyvy422,		core::raw_format::vyuy422)
		ADD_CONVERSION(core::raw_format::vyuy422, 		core::raw_format::uyvy422)
		ADD_CONVERSION(core::raw_format::yuyv422, 		core::raw_format::yvyu422)
		ADD_CONVERSION(core::raw_format::yvyu422,		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::uyvy422,		core::raw_format::yvyu422)
		ADD_CONVERSION(core::raw_format::vyuy422, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::yuyv422,		core::raw_format::vyuy422)
		ADD_CONVERSION(core::raw_format::yvyu422,		core::raw_format::uyvy422)

		ADD_CONVERSION(core::raw_format::yuyv422, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::yuv444, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::uyvy422, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::yuv444, 		core::raw_format::uyvy422)

		ADD_CONVERSION(core::raw_format::yuva4444, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::ayuv4444, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::yuv444, 		core::raw_format::yuva4444)
		ADD_CONVERSION(core::raw_format::yuv444, 		core::raw_format::ayuv4444)

		ADD_CONVERSION(core::raw_format::yuva4444, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::ayuv4444, 		core::raw_format::yuyv422)

		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::yuv444)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::yuv444)

		ADD_CONVERSION(core::raw_format::rgb24, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::rgba32, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::argb32, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::bgr24, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::bgra32, 		core::raw_format::yuyv422)
		ADD_CONVERSION(core::raw_format::abgr32, 		core::raw_format::yuyv422)

		ADD_CONVERSION(core::raw_format::yuv444,		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::yuyv422,		core::raw_format::rgb24)
		ADD_CONVERSION(core::raw_format::uyvy422,		core::raw_format::rgb24)


//		ADD_CONVERSION(core::raw_format::yuv422_v210, core::raw_format::yuyv422)

//		ADD_CONVERSION(YURI_FMT_V210, 		core::raw_format::yuyv422)
//		ADD_CONVERSION(YURI_FMT_V210, 		core::raw_format::uyvy422)
//		ADD_CONVERSION(core::raw_format::yuyv422,		YURI_FMT_V210)
//		ADD_CONVERSION(core::raw_format::uyvy422,	YURI_FMT_V210)

		ADD_CONVERSION(core::raw_format::u8, 	core::raw_format::y8)
		ADD_CONVERSION(core::raw_format::v8, 	core::raw_format::y8)
		ADD_CONVERSION(core::raw_format::r8, 	core::raw_format::y8)
		ADD_CONVERSION(core::raw_format::g8, 	core::raw_format::y8)
		ADD_CONVERSION(core::raw_format::b8, 	core::raw_format::y8)
		ADD_CONVERSION(core::raw_format::depth8, 	core::raw_format::y8)


	};

}

core::Parameters YuriConvertor::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p["colorimetry"]["Colorimetry to use when converting from RGB (BT709, BT601, BT2020)"]="BT709";
	p["format"]["Output format"]=std::string("YUV422");
	p["full"]["Assume YUV values in full range"]=true;
	return p;
}

YuriConvertor::YuriConvertor(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters)
	:core::SpecializedIOFilter<core::RawVideoFrame> (log_,parent,"YuriConv"),colorimetry_(YURI_COLORIMETRY_REC709),full_range_(true)
{
	IOTHREAD_INIT(parameters)
		log[log::info] << "Initialized " << converters.size() << " converters";
	for (auto it = converters.begin();	it!=converters.end(); ++it) {
		const format_pair_t& fp = it->first;
		log[log::debug] << "Converter: " << core::raw_format::get_format_name(fp.first) << " -> "
				<< core::raw_format::get_format_name(fp.second);
	}
}

YuriConvertor::~YuriConvertor() noexcept{
}

core::pFrame YuriConvertor::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	core::pRawVideoFrame outframe;
	core::pRawVideoFrame frame= dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) return outframe;

	format_t in_fmt = frame->get_format();
	format_pair_t conv_pair = std::make_pair(in_fmt, target_format);
	converter_t converter;

	if (converters.count(conv_pair)) converter = converters[conv_pair];
	if (converter) {
		outframe = converter(frame, *this);
		outframe->set_duration(input_frame->get_duration());
		outframe->set_timestamp(input_frame->get_timestamp());
	} else if (in_fmt == target_format) {
		outframe = frame;
	} else {

		log[log::debug] << "Unknown format combination " << core::raw_format::get_format_name(frame->get_format()) << " -> "
				<< core::raw_format::get_format_name(target_format) << "\n";
		return outframe;
	}

	if (outframe) {
		//@ TODO fix this...
		//outframe->set_info(frame->get_info());
		//if (outframe->get_pts() == 0) outframe->set_time(frame->get_pts(), frame->get_dts(), frame->get_duration());
	}
	return outframe;
}

core::pFrame YuriConvertor::do_special_single_step(const core::pRawVideoFrame& frame)
{

	return convert_frame(frame, format_);
}


bool YuriConvertor::set_param(const core::Parameter &p)
{
	if (iequals(p.get_name(),"colorimetry")) {
		std::string clr = p.get<std::string>();
		if (iequals(clr,"BT709") || iequals(clr,"REC709") || iequals(clr,"BT.709") || iequals(clr,"REC.709")) {
			colorimetry_=YURI_COLORIMETRY_REC709;
		} else if (iequals(clr,"BT601") || iequals(clr,"REC601") || iequals(clr,"BT.601") || iequals(clr,"REC.601")) {
			colorimetry_=YURI_COLORIMETRY_REC601;
		} else if (iequals(clr,"BT2020") || iequals(clr,"REC2020") || iequals(clr,"BT.2020") || iequals(clr,"REC.2020")) {
			colorimetry_=YURI_COLORIMETRY_REC2020;
		} else {
			log[log::warning] << "Unrecognized colorimetry type " << clr << ". Falling back to REC.709" << std::endl;
			colorimetry_=YURI_COLORIMETRY_REC709;
		}
	} else if (iequals(p.get_name(),"format")) {
		format_ = core::raw_format::parse_format(p.get<std::string>());
//		format_ = core::BasicPipe::get_format_from_string(p.get<std::string>(),YURI_TYPE_VIDEO);
		if (format_==core::raw_format::unknown) format_= core::raw_format::yuyv422;
		log[log::info] << "Output format " << core::raw_format::get_format_name(format_);
	} else if (iequals(p.get_name(),"full")) {
		full_range_ = p.get<bool>();
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(p);
	return true;
}

}

}




