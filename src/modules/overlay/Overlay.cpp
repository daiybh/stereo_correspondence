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
	return p;
}


Overlay::Overlay(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,2,1,std::string("overlay"))
{
	IO_THREAD_INIT("overlay")
}

Overlay::~Overlay()
{
}

namespace {
template <format_t f1, format_t f2>
struct combine_kernel {
void operator()(plane_t::const_iterator& src_pix, plane_t::const_iterator& ovr_pix, plane_t::iterator& dest_pix);
};

template<>
struct combine_kernel<YURI_FMT_RGBA, YURI_FMT_RGBA> {
	enum { src_bpp = 4 };
	enum { ovr_bpp = 4 };
	static format_t output_format() { return YURI_FMT_RGBA; }
void operator()
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
struct combine_kernel<YURI_FMT_RGB, YURI_FMT_RGBA> {
	enum { src_bpp = 3 };
	enum { ovr_bpp = 4 };
	static format_t output_format() { return YURI_FMT_RGBA; }
void operator()
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
struct combine_kernel<YURI_FMT_BGR, YURI_FMT_RGBA> {
	enum { src_bpp = 3 };
	enum { ovr_bpp = 4 };
	static format_t output_format() { return YURI_FMT_BGRA; }
void operator()
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
struct combine_kernel<YURI_FMT_BGRA, YURI_FMT_RGBA> {
	enum { src_bpp = 4 };
	enum { ovr_bpp = 4 };
	static format_t output_format() { return YURI_FMT_BGRA; }
void operator()
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
template<class kernel>
core::pBasicFrame combine(const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	const size_t 			width 		= frame_0->get_width();
	const size_t 			height 		= frame_0->get_height();
	const size_t 			w 			= frame_1->get_width();
	const size_t 			h 			= frame_1->get_height();
	const size_t 			linesize_0 	= width * kernel::src_bpp;
	const size_t 			linesize_1	= w     * kernel::ovr_bpp;
	const size_t 			linesize_out= w     * 4;
	core::pBasicFrame 		outframe 	= core::BasicIOThread::allocate_empty_frame(kernel::output_format(), width, height);
	const plane_t::const_iterator src 		= PLANE_DATA(frame_0,0).begin();
	const plane_t::const_iterator overlay 	= PLANE_DATA(frame_1,0).begin();
	const plane_t::iterator 	  dest 		= PLANE_DATA(outframe,0).begin();
	for (size_t line = 0; line < std::min(height,h); ++line) {
		plane_t::const_iterator src_pix 	= src+line*linesize_0;
		plane_t::const_iterator ovr_pix 	= overlay+line*linesize_1;
		plane_t::iterator 		dest_pix	= dest+line*linesize_out;
		for (size_t pixel = 0; pixel < std::min(width,w); ++pixel) {
			kernel()(src_pix, ovr_pix, dest_pix);
		}
	}
	return outframe;
}
template<format_t f>
core::pBasicFrame dispatch2(const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	format_t fmt = frame_1->get_format();
	switch (fmt) {
		case YURI_FMT_RGBA:
			return combine<combine_kernel<f, YURI_FMT_RGBA> >(frame_0, frame_1);
		case YURI_FMT_BGRA:
			return combine<combine_kernel<f, YURI_FMT_BGRA> >(frame_0, frame_1);
		default:
			break;
	}
	return core::pBasicFrame();
}

core::pBasicFrame dispatch(const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1)
{
	format_t fmt = frame_0->get_format();
	switch (fmt) {
		case YURI_FMT_RGB:
			return dispatch2<YURI_FMT_RGB>(frame_0, frame_1);
		case YURI_FMT_RGBA:
			return dispatch2<YURI_FMT_RGBA>(frame_0, frame_1);
		case YURI_FMT_BGR:
			return dispatch2<YURI_FMT_BGR>(frame_0, frame_1);
		case YURI_FMT_BGRA:
			return dispatch2<YURI_FMT_BGRA>(frame_0, frame_1);
		default:
			return core::pBasicFrame();
	}

}
}
bool Overlay::step()
{
	if (!frame_0) frame_0 = in[0]->pop_frame();
	if (!frame_1) frame_1 = in[1]->pop_frame();
	if (!frame_0 || !frame_1) {
		return true;
	}
	/*if (frame_0->get_format() != YURI_FMT_RGBA) {
		frame_0.reset();
		return true;
	}
	if (frame_1->get_format() != YURI_FMT_RGBA) {
		frame_1.reset();
		return true;
	}*/
	core::pBasicFrame outframe = dispatch(frame_0, frame_1);
	frame_0.reset();
	frame_1.reset();
	if (outframe) push_raw_video_frame(0, outframe);
	return true;
}
bool Overlay::set_param(const core::Parameter& param)
{
	return core::BasicIOThread::set_param(param);
}

} /* namespace overlay */
} /* namespace yuri */


