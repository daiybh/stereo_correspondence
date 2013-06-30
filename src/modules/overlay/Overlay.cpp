/*!
 * @file 		Overlay.cpp
 * @author 		<Your name>
 * @date		27.05.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Overlay.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace overlay {

REGISTER("overlay",Overlay)

IO_THREAD_GENERATOR(Overlay)

core::pParameters Overlay::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Overlay");
	p->set_max_pipes(1,1);
	(*p)["x"]["X offset"]=0;
	(*p)["y"]["Y offset"]=0;
	return p;
}


Overlay::Overlay(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicMultiIOFilter(log_,parent,2,1,std::string("overlay"))
{
	IO_THREAD_INIT("overlay")
}

Overlay::~Overlay()
{
}

namespace {
template<size_t s, size_t o, size_t d, format_t fmt>
struct combine_base {
	enum { src_bpp 	= s };
	enum { ovr_bpp 	= o };
	enum { dest_bpp	= d };
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
struct combine_kernel<YURI_FMT_RGBA, YURI_FMT_RGBA>:
public combine_base<4, 4, 4, YURI_FMT_RGBA> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + 255 * alpha);
	ovr_pix++;
}

};
template<>
struct combine_kernel<YURI_FMT_RGB, YURI_FMT_RGBA>:
public combine_base<3, 4, 4, YURI_FMT_RGBA>
{
	static void compute
	(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
	{
		const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
		const double minus_alpha = 1.0 - alpha;
		*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *ovr_pix++ * alpha);
		*dest_pix++ =static_cast<ubyte_t>(255*minus_alpha + 255 * alpha);
		ovr_pix++;
	}
};

template<>
struct combine_kernel<YURI_FMT_BGR, YURI_FMT_RGBA>:
public combine_base<3, 4, 4, YURI_FMT_BGRA> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+2) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+1) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+0) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(255*minus_alpha + 255 * alpha);
	ovr_pix+=4;
}
};
template<>
struct combine_kernel<YURI_FMT_BGRA, YURI_FMT_RGBA>:
public combine_base<4, 4, 4, YURI_FMT_BGRA> {
	static void compute
(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	const double alpha = static_cast<double>(*(ovr_pix+3))/255.0;
	const double minus_alpha = 1.0 - alpha;
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+2) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+1) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + *(ovr_pix+0) * alpha);
	*dest_pix++ =static_cast<ubyte_t>(*src_pix++*minus_alpha + 255 * alpha);
	ovr_pix+=4;
}
};

template<>
struct combine_kernel<YURI_FMT_BGR, YURI_FMT_BGRA>:
public combine_kernel<YURI_FMT_RGB, YURI_FMT_RGBA>{
	static format_t output_format() { return YURI_FMT_BGRA; }
};

template<>
struct combine_kernel<YURI_FMT_BGRA, YURI_FMT_BGRA>:
public combine_kernel<YURI_FMT_RGBA, YURI_FMT_RGBA>{
	static format_t output_format() { return YURI_FMT_BGRA; }
};
template<>
struct combine_kernel<YURI_FMT_RGB, YURI_FMT_BGRA>:
public combine_kernel<YURI_FMT_BGR, YURI_FMT_RGBA>{
	static format_t output_format() { return YURI_FMT_RGBA; }
};
template<>
struct combine_kernel<YURI_FMT_RGBA, YURI_FMT_BGRA>:
public combine_kernel<YURI_FMT_BGRA, YURI_FMT_RGBA>{
	static format_t output_format() { return YURI_FMT_RGBA; }
};


template<>
struct combine_kernel<YURI_FMT_RGBA, YURI_FMT_RGB>:
public combine_base<4, 3, 4, YURI_FMT_RGBA> {
	static void compute
(plane_t::const_iterator& , plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =*ovr_pix++;
	*dest_pix++ =255;
}

};
template<>
struct combine_kernel<YURI_FMT_RGB, YURI_FMT_RGB>:
public combine_base<3, 3, 3, YURI_FMT_RGB>
{
	static void compute
	(plane_t::const_iterator& , plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
	{
		*dest_pix++ =*ovr_pix++;
		*dest_pix++ =*ovr_pix++;
		*dest_pix++ =*ovr_pix++;
	}
};

template<>
struct combine_kernel<YURI_FMT_BGR, YURI_FMT_RGB>:
public combine_base<3, 3, 3, YURI_FMT_BGR> {
	static void compute
(plane_t::const_iterator& , plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*(ovr_pix+2);
	*dest_pix++ =*(ovr_pix+1);
	*dest_pix++ =*(ovr_pix+0);
	ovr_pix+=3;
}
};
template<>
struct combine_kernel<YURI_FMT_BGRA, YURI_FMT_RGB>:
public combine_base<4, 3, 4, YURI_FMT_BGRA> {
	static void compute
(plane_t::const_iterator& , plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix)
{
	*dest_pix++ =*(ovr_pix+2);
	*dest_pix++ =*(ovr_pix+1);
	*dest_pix++ =*(ovr_pix+0);
	*dest_pix++ =255;
	ovr_pix+=3;
}
};

template<>
struct combine_kernel<YURI_FMT_BGR, YURI_FMT_BGR>:
public combine_kernel<YURI_FMT_RGB, YURI_FMT_RGB>{
	static format_t output_format() { return YURI_FMT_BGR; }
};

template<>
struct combine_kernel<YURI_FMT_BGRA, YURI_FMT_BGR>:
public combine_kernel<YURI_FMT_RGBA, YURI_FMT_RGB>{
	static format_t output_format() { return YURI_FMT_BGRA; }
};
template<>
struct combine_kernel<YURI_FMT_RGB, YURI_FMT_BGR>:
public combine_kernel<YURI_FMT_BGR, YURI_FMT_RGB>{
	static format_t output_format() { return YURI_FMT_RGB; }
};
template<>
struct combine_kernel<YURI_FMT_RGBA, YURI_FMT_BGR>:
public combine_kernel<YURI_FMT_BGRA, YURI_FMT_RGB>{
	static format_t output_format() { return YURI_FMT_RGBA; }
};
template<format_t f>
core::pBasicFrame dispatch2(Overlay& overlay, const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	format_t fmt = frame_1->get_format();
	switch (fmt) {
		case YURI_FMT_RGBA:
			return overlay.combine<combine_kernel<f, YURI_FMT_RGBA> >(frame_0, frame_1);
		case YURI_FMT_BGRA:
			return overlay.combine<combine_kernel<f, YURI_FMT_BGRA> >(frame_0, frame_1);
		case YURI_FMT_RGB:
			return overlay.combine<combine_kernel<f, YURI_FMT_RGB> >(frame_0, frame_1);
		case YURI_FMT_BGR:
			return overlay.combine<combine_kernel<f, YURI_FMT_BGR> >(frame_0, frame_1);
		default:
			break;
	}
	return core::pBasicFrame();
}

core::pBasicFrame dispatch(Overlay& overlay, const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	format_t fmt = frame_0->get_format();
	switch (fmt) {
		case YURI_FMT_RGB:
			return dispatch2<YURI_FMT_RGB>(overlay, frame_0, frame_1);
		case YURI_FMT_RGBA:
			return dispatch2<YURI_FMT_RGBA>(overlay, frame_0, frame_1);
		case YURI_FMT_BGR:
			return dispatch2<YURI_FMT_BGR>(overlay, frame_0, frame_1);
		case YURI_FMT_BGRA:
			return dispatch2<YURI_FMT_BGRA>(overlay, frame_0, frame_1);
		default:
			return core::pBasicFrame();
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
core::pBasicFrame Overlay::combine(const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	const ssize_t 			width 		= frame_0->get_width();
	const ssize_t 			height 		= frame_0->get_height();
	const ssize_t 			w 			= frame_1->get_width();
	const ssize_t 			h 			= frame_1->get_height();
	const size_t 			linesize_0 	= width * kernel::src_bpp;
	const size_t 			linesize_1	= w     * kernel::ovr_bpp;
	const size_t 			linesize_out= width * kernel::dest_bpp;;
	log[log::verbose_debug] << "Base " << width << "x" << height << " (" << linesize_0 << ") + " << w << "x" <<h << " (" << linesize_1 << ") -> ("<<linesize_out<<")";
	core::pBasicFrame 		outframe 	= core::BasicIOThread::allocate_empty_frame(kernel::output_format(), width, height);
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
		if (x_ > 0) {
			fill_line<kernel>(pixel, std::min(width,x_), src_pix, dest_pix);
		}
		for (; pixel < std::min(width,w+x_); ++pixel) {
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
std::vector<core::pBasicFrame> Overlay::do_single_step(const std::vector<core::pBasicFrame>& frames)
{
	assert(frames.size() == 2);
	core::pBasicFrame outframe = dispatch(*this, frames[0], frames[1]);
	if (outframe) return {outframe};
	return {};
}
bool Overlay::set_param(const core::Parameter& param)
{
	using boost::iequals;
	if (iequals(param.name,"x")) {
		x_ = param.get<ssize_t>();
	} else if (iequals(param.name,"y")) {
		y_ = param.get<ssize_t>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace overlay */
} /* namespace yuri */


