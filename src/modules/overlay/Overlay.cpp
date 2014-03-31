/*!
 * @file 		Overlay.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		27.05.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Overlay.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"
//#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <cassert>
namespace yuri {
namespace overlay {
using plane_t = core::RawVideoFrame::value_type;
IOTHREAD_GENERATOR(Overlay)

MODULE_REGISTRATION_BEGIN("overlay")
		REGISTER_IOTHREAD("overlay",Overlay)
MODULE_REGISTRATION_END()

core::Parameters Overlay::configure()
{
	core::Parameters p = core::MultiIOFilter::configure();
	p.set_description("Overlay");
//	p->set_max_pipes(1,1);
	p["x"]["X offset"]=0;
	p["y"]["Y offset"]=0;
	return p;
}


Overlay::Overlay(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
		SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_,parent,1,std::string("overlay")),
event::BasicEventConsumer(log)
{
	IOTHREAD_INIT(parameters)
}

Overlay::~Overlay() noexcept
{
}

namespace {
using namespace core::raw_format;
/*!
 * @tparam	s	- bytes per input (background image) pixel
 * @tparam	o	- bytes overlay image pixel
 * @tparam	d	- bytes per destination pixel
 * @tparam	fmt	- format of destination image
 * @tparam	step- number of pixel the kernel copies at once
 */
template<size_t s, size_t o, size_t d, format_t fmt, size_t step = 1>
struct combine_base {
	enum { src_bpp 	= s };
	enum { ovr_bpp 	= o };
	enum { dest_bpp	= d };
	enum { pix_step	= step };
	static format_t output_format() { return fmt; }

	static void fill(plane_t::const_iterator& src_pix, plane_t::iterator& dest_pix) {
		size_t i = 0;
		for (; i < s; ++i) {
			*dest_pix++ = *src_pix++;
		}
		for (; i < d; ++i) {
			*dest_pix++ = 255;
		}
	}
};

template <format_t f1, format_t f2>
struct combine_kernel {
void operator()(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix);
};

template<>
struct combine_kernel<rgba32, rgba32>:
public combine_base<4, 4, 4, rgba32> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + 255 * alpha);
	ovr_pix++;
}

};
template<>
struct combine_kernel<rgb24, rgba32>:
public combine_base<3, 4, 4, rgba32>
{
	static void compute
	(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
	{
		const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
		const double minus_alpha = 1.0 - alpha;
		*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<uint8_t>(255*minus_alpha + 255 * alpha);
		ovr_pix++;
	}
};

template<>
struct combine_kernel<bgr24, rgba32>:
public combine_base<3, 4, 4, abgr32> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+2) * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+1) * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+0) * alpha);
	*dest_pix++ =static_cast<uint8_t>(255*minus_alpha + 255 * alpha);
	ovr_pix+=4;
}
};
template<>
struct combine_kernel<abgr32, rgba32>:
public combine_base<4, 4, 4, abgr32> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+2) * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+1) * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + *(ovr_pix+0) * alpha);
	*dest_pix++ =static_cast<uint8_t>(*src_pix++*minus_alpha + 255 * alpha);
	ovr_pix+=4;
}
};

template<>
struct combine_kernel<bgr24, abgr32>:
public combine_kernel<rgb24, rgba32>{
	static format_t output_format() { return abgr32; }
};

template<>
struct combine_kernel<abgr32, abgr32>:
public combine_kernel<rgba32, rgba32>{
	static format_t output_format() { return abgr32; }
};
template<>
struct combine_kernel<rgb24, abgr32>:
public combine_kernel<bgr24, rgba32>{
	static format_t output_format() { return rgba32; }
};
template<>
struct combine_kernel<rgba32, abgr32>:
public combine_kernel<abgr32, rgba32>{
	static format_t output_format() { return rgba32; }
};


template<>
struct combine_kernel<rgba32, rgb24>:
public combine_base<4, 3, 4, rgba32> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =255;
	src_pix+=4;
}

};
template<>
struct combine_kernel<rgb24, rgb24>:
public combine_base<3, 3, 3, rgb24>
{
	static void compute
	(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
	{
		*dest_pix++ =*ovr_pix++;
		*dest_pix++ =*ovr_pix++;
		*dest_pix++ =*ovr_pix++;
		src_pix+=3;
	}
};

template<>
struct combine_kernel<bgr24, rgb24>:
public combine_base<3, 3, 3, bgr24> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*(ovr_pix+2);
	*dest_pix++ =*(ovr_pix+1);
	*dest_pix++ =*(ovr_pix+0);
	ovr_pix+=3;
	src_pix+=3;
}
};
template<>
struct combine_kernel<abgr32, rgb24>:
public combine_base<4, 3, 4, abgr32> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*(ovr_pix+2);
	*dest_pix++ =*(ovr_pix+1);
	*dest_pix++ =*(ovr_pix+0);
	*dest_pix++ =255;
	ovr_pix+=3;
	src_pix+=4;
}
};

template<>
struct combine_kernel<bgr24, bgr24>:
public combine_kernel<rgb24, rgb24>{
	static format_t output_format() { return bgr24; }
};

template<>
struct combine_kernel<abgr32, bgr24>:
public combine_kernel<rgba32, rgb24>{
	static format_t output_format() { return abgr32; }
};
template<>
struct combine_kernel<rgb24, bgr24>:
public combine_kernel<bgr24, rgb24>{
	static format_t output_format() { return rgb24; }
};
template<>
struct combine_kernel<rgba32, bgr24>:
public combine_kernel<abgr32, rgb24>{
	static format_t output_format() { return rgba32; }
};

template<>
struct combine_kernel<yuyv422, yuyv422>:
public combine_base<2, 2, 2, yuyv422, 2> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	src_pix+=4;
}
};
template<>
struct combine_kernel<yuyv422, yuv444>:
public combine_base<2, 3, 3, yuv444, 2> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	std::copy(ovr_pix, ovr_pix+6, dest_pix);
	ovr_pix+=6;
	dest_pix+=6;
	src_pix+=4;
//	*dest_pix++ =*(ovr_pix+0);
//	*dest_pix++ =*(ovr_pix+1);
//	*dest_pix++ =*(ovr_pix+3);
//	*dest_pix++ =*(ovr_pix+2);
//	*dest_pix++ =*(ovr_pix+1);
//	*dest_pix++ =*(ovr_pix+3);
//	ovr_pix+=4;
}
};

template<>
struct combine_kernel<yuyv422, yuva4444>:
public combine_base<2, 4, 4, yuva4444, 2> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = ovr_pix[3]/255.0;
	const double minus_alpha = 1.0 - alpha;

//	std::fill(dest_pix, dest_pix+4, 0);

	*dest_pix++ =static_cast<uint8_t>((minus_alpha * *src_pix++) + (*ovr_pix++ * alpha));
	const uint8_t u = *src_pix++;
	*dest_pix++ =static_cast<uint8_t>((minus_alpha * u) + (*ovr_pix++ * alpha));
	const uint8_t y2 = *src_pix++;
	const uint8_t v = *src_pix++;
	*dest_pix++ =static_cast<uint8_t>((minus_alpha * v) + (*ovr_pix++ * alpha));
	*dest_pix++ = 255;
	ovr_pix++;

	*dest_pix++ =static_cast<uint8_t>((minus_alpha * y2) + (*ovr_pix++ * alpha));
	*dest_pix++ =static_cast<uint8_t>((minus_alpha * u) + (*ovr_pix++ * alpha));
	*dest_pix++ =static_cast<uint8_t>((minus_alpha * v) + (*ovr_pix++ * alpha));
	*dest_pix++ = 255;
	ovr_pix++;
}
};
template<format_t f>
core::pRawVideoFrame dispatch2(Overlay& overlay, const core::pRawVideoFrame& frame_0, const core::pRawVideoFrame& frame_1)
{
	format_t fmt = frame_1->get_format();
	switch (fmt) {
		case rgba32:
			return overlay.combine<combine_kernel<f, rgba32> >(frame_0, frame_1);
		case abgr32:
			return overlay.combine<combine_kernel<f, abgr32> >(frame_0, frame_1);
		case rgb24:
			return overlay.combine<combine_kernel<f, rgb24> >(frame_0, frame_1);
		case bgr24:
			return overlay.combine<combine_kernel<f, bgr24> >(frame_0, frame_1);
		default:
			break;
	}
	return core::pRawVideoFrame();
}
template<format_t f>
core::pRawVideoFrame dispatch2_yuv(Overlay& overlay, const core::pRawVideoFrame& frame_0, const core::pRawVideoFrame& frame_1)
{
	format_t fmt = frame_1->get_format();
	switch (fmt) {
		case yuyv422:
			return overlay.combine<combine_kernel<f, yuyv422> >(frame_0, frame_1);
		case yuv444:
			return overlay.combine<combine_kernel<f, yuv444> >(frame_0, frame_1);
		case yuva4444:
			return overlay.combine<combine_kernel<f, yuva4444> >(frame_0, frame_1);
		default:
			break;
	}
	return core::pRawVideoFrame();
}

core::pRawVideoFrame dispatch(Overlay& overlay, const core::pRawVideoFrame& frame_0, const core::pRawVideoFrame& frame_1)
{
	format_t fmt = frame_0->get_format();
	switch (fmt) {
		case rgb24:
			return dispatch2<rgb24>(overlay, frame_0, frame_1);
		case rgba32:
			return dispatch2<rgba32>(overlay, frame_0, frame_1);
		case bgr24:
			return dispatch2<bgr24>(overlay, frame_0, frame_1);
		case abgr32:
			return dispatch2<abgr32>(overlay, frame_0, frame_1);
		case yuyv422:
			return dispatch2_yuv<yuyv422>(overlay, frame_0, frame_1);
		default:
			return core::pRawVideoFrame();
	}

}
template<class kernel>
inline void fill_line(ssize_t& pixel, const ssize_t& max_pixel, plane_t::const_iterator& src_pix, plane_t::iterator& dest_pix)
{
	for (; pixel < max_pixel; ++pixel) {
		kernel::fill(src_pix, dest_pix);
	}
}
template<class kernel>
inline void fill_multiple_lines(ssize_t& line, const ssize_t& max_line,
		const plane_t::const_iterator& src, const plane_t::iterator& dest,
		const size_t linesize_0, const size_t linesize_out, const size_t width)
{
	for (; line < max_line; ++line) {
		plane_t::const_iterator src_pix 	= src+line*linesize_0;
		plane_t::iterator 		dest_pix	= dest+line*linesize_out;
		ssize_t pixel = 0;
		fill_line<kernel>(pixel, width, src_pix, dest_pix);
	}
}
}
template<class kernel>
core::pRawVideoFrame Overlay::combine(const core::pRawVideoFrame& frame_0, const core::pRawVideoFrame& frame_1)
{
	const resolution_t		res_0		= frame_0->get_resolution();
	const resolution_t		res_1		= frame_1->get_resolution();
	const ssize_t 			width 		= res_0.width;
	const ssize_t 			height 		= res_0.height;
	const ssize_t 			w 			= res_1.width;
	const ssize_t 			h 			= res_1.height;
	const size_t 			linesize_0 	= width * kernel::src_bpp;
	const size_t 			linesize_1	= w     * kernel::ovr_bpp;
	const size_t 			linesize_out= width * kernel::dest_bpp;
	const size_t			step		= kernel::pix_step;
	const ssize_t			x			= x_ - (x_ % step);

	log[log::verbose_debug] << "Base " << width << "x" << height << " (" << linesize_0 << ") + " << w << "x" <<h << " (" << linesize_1 << ") -> ("<<linesize_out<<")";
	core::pRawVideoFrame 		outframe 	= core::RawVideoFrame::create_empty(kernel::output_format(), res_0);
	const plane_t::const_iterator src 		= PLANE_DATA(frame_0,0).begin();
	const plane_t::const_iterator overlay 	= PLANE_DATA(frame_1,0).begin();
	const plane_t::iterator 	  dest 		= PLANE_DATA(outframe,0).begin();
	ssize_t line = 0;
	if (y_ > 0) {
		fill_multiple_lines<kernel>(line,std::min(height,y_), src, dest, linesize_0, linesize_out, width);
	}
	for (; line < std::min(height,h+y_); ++line) {
		plane_t::const_iterator src_pix 	= src+line*linesize_0;
		plane_t::const_iterator ovr_pix 	= overlay+(line-y_)*linesize_1;
		plane_t::iterator 		dest_pix	= dest+line*linesize_out;
		ssize_t pixel = 0;
		if (x > 0) {
			fill_line<kernel>(pixel, std::min(width,x), src_pix, dest_pix);
		}
		for (; pixel < std::min(width,w+x); pixel+=step) {
			kernel::compute(src_pix, ovr_pix, dest_pix);
		}
		if (pixel < width-1) {
			fill_line<kernel>(pixel, width, src_pix, dest_pix);
		}
	}
	if (line < height - 1) {
		fill_multiple_lines<kernel>(line, height, src, dest, linesize_0, linesize_out, width);
	}
	return outframe;
}
std::vector<core::pFrame> Overlay::do_special_step(const param_type& frames)
{
	process_events();
	core::pRawVideoFrame f0 = std::get<0>(frames);
	core::pRawVideoFrame f1 = std::get<1>(frames);
	if (!f0 || !f1) return {};
	core::pRawVideoFrame outframe = dispatch(*this, f0, f1);
	if (outframe) return {outframe};
	return {};
}
bool Overlay::set_param(const core::Parameter& param)
{
//	using boost::iequals;
	if (iequals(param.get_name(),"x")) {
		x_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"y")) {
		y_ = param.get<ssize_t>();
	} else return core::MultiIOFilter::set_param(param);
	return true;
}

bool Overlay::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	log[log::info] << "Received event " << event_name;
	try {
		using event::get_value;
	if (event_name == "x") {
		x_ = get_value<event::EventInt>(event);
	} else if (event_name == "y") {
		y_ = get_value<event::EventInt>(event);
	} else return false;
	}
	catch (std::bad_cast&) {
		log[log::info] << "bad cast in " << event_name;
		return false;
	}
	return true;
}
} /* namespace overlay */
} /* namespace yuri */


