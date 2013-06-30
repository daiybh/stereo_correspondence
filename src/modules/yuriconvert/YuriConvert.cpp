/*!
 * @file 		YuriConvertor.cpp
 * @author 		Zdenek Travnicek
 * @date 		13.8.2010
 * @date		16.2.2013
 * * @date		26.5.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "YuriConvert.h"
#include "yuri/core/Module.h"
#include <boost/assign.hpp>
#include <boost/function.hpp>


namespace yuri {

namespace video {

REGISTER("yuri_convert",YuriConvertor)
IO_THREAD_GENERATOR(YuriConvertor)


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


	typedef boost::function<core::pBasicFrame (const core::pBasicFrame&, const YuriConvertor&)> converter_t;
	typedef std::pair<yuri::format_t, yuri::format_t> format_pair_t;

//	inline unsigned int convY(unsigned int Y) { return (Y*219+4128) >> 6; }
//	inline unsigned int convC(unsigned int C) {	return (C*7+129) >> 1; }

	template<format_t fmt>
		core::pBasicFrame allocate_frame(size_t width, size_t height);
	template<format_t fmt>
		size_t get_linesize(size_t width);
	template<format_t fmt_in, format_t fmt_out>
		void convert_line(plane_t::const_iterator src, plane_t::iterator dest, size_t width, const YuriConvertor&);
	template<format_t fmt_in, format_t fmt_out>
		void convert_line(plane_t::const_iterator src, plane_t::iterator dest, size_t width);
	template<format_t fmt_in, format_t fmt_out>
		core::pBasicFrame convert_formats( const core::pBasicFrame& frame, const YuriConvertor&);

	#include "YuriConvert.impl"

#define ADD_CONVERSION(fmt1, fmt2) (std::make_pair(fmt1, fmt2), &convert_formats<fmt1, fmt2>)

	std::map<format_pair_t, converter_t> converters =
		boost::assign::map_list_of<format_pair_t, converter_t>
		ADD_CONVERSION(YURI_FMT_RGB24, 		YURI_FMT_RGBA)
		ADD_CONVERSION(YURI_FMT_BGR, 		YURI_FMT_BGRA)
		ADD_CONVERSION(YURI_FMT_RGBA, 		YURI_FMT_RGB24)
		ADD_CONVERSION(YURI_FMT_BGRA, 		YURI_FMT_BGR)

		ADD_CONVERSION(YURI_FMT_BGRA, 		YURI_FMT_RGBA)
		ADD_CONVERSION(YURI_FMT_RGBA, 		YURI_FMT_BGRA)
		ADD_CONVERSION(YURI_FMT_BGR, 		YURI_FMT_RGB)
		ADD_CONVERSION(YURI_FMT_RGB24, 		YURI_FMT_BGR)

		ADD_CONVERSION(YURI_FMT_BGRA, 		YURI_FMT_RGB)
		ADD_CONVERSION(YURI_FMT_RGBA, 		YURI_FMT_BGR)
		ADD_CONVERSION(YURI_FMT_BGR, 		YURI_FMT_RGBA)
		ADD_CONVERSION(YURI_FMT_RGB24, 		YURI_FMT_BGRA)


		ADD_CONVERSION(YURI_FMT_YUV422, 	YURI_FMT_UYVY422)
		ADD_CONVERSION(YURI_FMT_UYVY422, 	YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_YVYU422,	YURI_FMT_VYUY422)
		ADD_CONVERSION(YURI_FMT_VYUY422,	YURI_FMT_YVYU422)
		ADD_CONVERSION(YURI_FMT_UYVY422,	YURI_FMT_VYUY422)
		ADD_CONVERSION(YURI_FMT_VYUY422, 	YURI_FMT_UYVY422)
		ADD_CONVERSION(YURI_FMT_YUV422, 	YURI_FMT_YVYU422)
		ADD_CONVERSION(YURI_FMT_YVYU422,	YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_UYVY422,	YURI_FMT_YVYU422)
		ADD_CONVERSION(YURI_FMT_VYUY422, 	YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_YUV422,		YURI_FMT_VYUY422)
		ADD_CONVERSION(YURI_FMT_YVYU422,	YURI_FMT_UYVY422)

		ADD_CONVERSION(YURI_FMT_YUV422, 	YURI_FMT_YUV444)
		ADD_CONVERSION(YURI_FMT_YUV444, 	YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_UYVY422, 	YURI_FMT_YUV444)
		ADD_CONVERSION(YURI_FMT_YUV444, 	YURI_FMT_UYVY422)

		ADD_CONVERSION(YURI_FMT_RGB24, 		YURI_FMT_YUV444)
		ADD_CONVERSION(YURI_FMT_RGBA, 		YURI_FMT_YUV444)
		ADD_CONVERSION(YURI_FMT_BGR, 		YURI_FMT_YUV444)
		ADD_CONVERSION(YURI_FMT_BGRA, 		YURI_FMT_YUV444)

		ADD_CONVERSION(YURI_FMT_RGB24, 		YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_RGBA, 		YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_BGR, 		YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_BGRA, 		YURI_FMT_YUV422)

		ADD_CONVERSION(YURI_FMT_YUV444,		YURI_FMT_RGB24)
		ADD_CONVERSION(YURI_FMT_YUV422,		YURI_FMT_RGB24)
		ADD_CONVERSION(YURI_FMT_UYVY422,	YURI_FMT_RGB24)

		ADD_CONVERSION(YURI_FMT_V210, 		YURI_FMT_YUV422)
		ADD_CONVERSION(YURI_FMT_V210, 		YURI_FMT_UYVY422)
		ADD_CONVERSION(YURI_FMT_YUV422,		YURI_FMT_V210)
		ADD_CONVERSION(YURI_FMT_UYVY422,	YURI_FMT_V210)

	;

}

core::pParameters YuriConvertor::configure()
{
	core::pParameters p = BasicIOFilter::configure();
	(*p)["colorimetry"]["Colorimetry to use when converting from RGB (BT709, BT601, BT2020)"]="BT709";
	(*p)["format"]["Output format"]=std::string("YUV422");
	(*p)["full"]["Assume YUV values in full range"]=true;
	return p;
}

YuriConvertor::YuriConvertor(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters) IO_THREAD_CONSTRUCTOR
	:core::BasicIOFilter(log_,parent,"YuriConv"),colorimetry_(YURI_COLORIMETRY_REC709),full_range_(true)
{
	IO_THREAD_INIT("Yuri Convert")
		log[log::info] << "Initialized " << converters.size() << " converters";
	for (std::map<format_pair_t, converter_t>::iterator it = converters.begin();
			it!=converters.end(); ++it) {
		const format_pair_t& fp = it->first;
		log[log::debug] << "Converter: " << core::BasicPipe::get_format_string(fp.first) << " -> "
				<< core::BasicPipe::get_format_string(fp.second);
	}
}

YuriConvertor::~YuriConvertor() {
}

core::pBasicFrame YuriConvertor::do_simple_single_step(const core::pBasicFrame& frame)
//bool YuriConvertor::step()
{
//	if (!in[0] || in[0]->is_empty()) return true;
//	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return core::pBasicFrame();
	core::pBasicFrame outframe;
	format_t in_fmt = frame->get_format();
	format_pair_t conv_pair = std::make_pair(in_fmt, format_);
	converter_t converter;

	if (converters.count(conv_pair)) converter = converters[conv_pair];
	if (converter) {
		outframe = converter(frame, *this);
	} else if (in_fmt == format_) {
		outframe = frame;
	} else {
		log[log::debug] << "Unknown format combination " << core::BasicPipe::get_format_string(frame->get_format()) << " -> "
				<< core::BasicPipe::get_format_string(format_) << "\n";
		return core::pBasicFrame();
	}
	if (outframe) {
		outframe->set_info(frame->get_info());
		if (outframe->get_pts() == 0) outframe->set_time(frame->get_pts(), frame->get_dts(), frame->get_duration());
		//push_raw_video_frame (0,outframe);
		return outframe;
	}
	return core::pBasicFrame();
}


bool YuriConvertor::set_param(const core::Parameter &p)
{
	using boost::iequals;
	if (iequals(p.name,"colorimetry")) {
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
	} else if (iequals(p.name,"format")) {
		format_ = core::BasicPipe::get_format_from_string(p.get<std::string>(),YURI_TYPE_VIDEO);
		if (format_==YURI_FMT_NONE) format_=YURI_FMT_YUV422;
		log[log::info] << "Output format " << core::BasicPipe::get_format_string(format_);
	} else if (iequals(p.name,"full")) {
		full_range_ = p.get<bool>();
	} else return BasicIOThread::set_param(p);
	return true;
}

}

}



