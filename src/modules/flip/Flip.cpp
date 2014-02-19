/*!
 * @file 		Flip.cpp
 * @author 		Zdenek Travnicek
 * @date 		16.3.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Flip.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <cassert>
#include <algorithm>
namespace yuri {

namespace io {

IOTHREAD_GENERATOR(Flip)

MODULE_REGISTRATION_BEGIN("flip")
		REGISTER_IOTHREAD("flip",Flip)
MODULE_REGISTRATION_END()

core::Parameters Flip::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p["flip_x"]["flip x (around y axis)"]=true;
	p["flip_y"]["flip y (around X axis)"]=false;
	//p->set_max_pipes(1,1);
	//p->add_input_format(YURI_FMT_RGB);
	//p->add_output_format(YURI_FMT_RGB);
	return p;
}


namespace {
bool verify_support(const core::raw_format::raw_format_t& fmt)
{
	if (fmt.planes.size() > 1) return false;
	if (fmt.planes.empty()) return false;
	const auto& plane= fmt.planes[0];
	if (plane.components.empty()) return false;
	const auto& depth = plane.bit_depth;
	if (depth.first % (depth.second * 8)) return false;

	return true;
}
}

Flip::Flip(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters)
:core::SpecializedIOFilter<core::RawVideoFrame>(_log,parent,"flip"),
 flip_x_(true),flip_y_(false)
 {
	IOTHREAD_INIT(parameters)
	std::vector<format_t> supported_fmts;
	for (const auto& f: core::raw_format::formats()) {
		if (verify_support(f.second)) {
			supported_fmts.push_back(f.first);
			log[log::verbose_debug] << "Setting format " << f.second.name << " as supported";
		}
	}
	set_supported_formats(supported_fmts);

}

Flip::~Flip() noexcept {

}


namespace {


template<int bpp>
struct cpy_helper {
	std::array<uint8_t, bpp> data;
};

struct cpy_helper_yuyv {
	std::array<uint8_t, 4> data;
	cpy_helper_yuyv& operator=(const cpy_helper_yuyv& rhs) {
		data[0]=rhs.data[2];
		data[1]=rhs.data[1];
		data[2]=rhs.data[0];
		data[3]=rhs.data[3];
		return *this;
	}
};
struct cpy_helper_uyvy {
	std::array<uint8_t, 4> data;
	cpy_helper_uyvy& operator=(const cpy_helper_uyvy& rhs) {
		data[0]=rhs.data[0];
		data[1]=rhs.data[3];
		data[2]=rhs.data[2];
		data[3]=rhs.data[1];
		return *this;
	}
};


template<class Iter, class Iter2>
	void flip_line(Iter src_begin, Iter src_end, Iter2 dst_end)
{
	for (;src_begin!=src_end;) {
		*(--dst_end)=*src_begin++;
	}
}

template<class T>
void flip_line_conv(const uint8_t* src, const uint8_t* src_end, uint8_t* dest)
{
	const T* s0 = reinterpret_cast<const T*>(src);
	const T* s1 = reinterpret_cast<const T*>(src_end);
	T* d0 = reinterpret_cast<T*>(dest);
	flip_line(s0, s1, d0);
}

void flip_dispatch(int yuv_pos, int bpp, const uint8_t* in_ptr, const uint8_t* in_ptr_end, uint8_t* out_ptr_end)
{
	if (yuv_pos == 0) {
		flip_line_conv<cpy_helper_yuyv>(in_ptr, in_ptr_end, out_ptr_end);
	} else if (yuv_pos == 1) {
		flip_line_conv<cpy_helper_uyvy>(in_ptr, in_ptr_end, out_ptr_end);
	} else switch (bpp) {
		case 1:flip_line_conv<cpy_helper<1>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 2:flip_line_conv<cpy_helper<2>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 3:flip_line_conv<cpy_helper<3>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 4:flip_line_conv<cpy_helper<4>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 5:flip_line_conv<cpy_helper<5>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 6:flip_line_conv<cpy_helper<6>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 7:flip_line_conv<cpy_helper<7>>(in_ptr, in_ptr_end, out_ptr_end);break;
		case 8:flip_line_conv<cpy_helper<8>>(in_ptr, in_ptr_end, out_ptr_end);break;
		default:break;
	}
}

}

core::pFrame Flip::do_special_single_step(const core::pRawVideoFrame& frame)
{
	if (!flip_x_ && !flip_y_) return frame;
	size_t	w = frame->get_width();
	size_t	h = frame->get_height();



	const auto& fi = core::raw_format::get_format_info(frame->get_format());

	if (!verify_support(fi)) return {};

	// multiple planes shoul be implemented as well..
//	assert(fi.planes.size() == 1);

	const auto depth = fi.planes[0].bit_depth;
	const size_t bpp = depth.first/depth.second/8;
	const size_t line_length = w*bpp;
	int yuv_y_pos = -1;

	if (frame->get_format()==core::raw_format::yuyv422 ||
		frame->get_format()==core::raw_format::yvyu422) {
		yuv_y_pos = 0;
	} else if (frame->get_format()==core::raw_format::uyvy422 ||
			frame->get_format()==core::raw_format::vyuy422 ) {
		yuv_y_pos = 1;
	}


	core::pRawVideoFrame  frame_out = core::RawVideoFrame::create_empty(frame->get_format(), frame->get_resolution());

	const size_t src_line_advance = flip_y_?-line_length:line_length;

	const uint8_t * base_ptr =  PLANE_RAW_DATA(frame,0);
	if (flip_y_) base_ptr += (h-1)*line_length;
	const uint8_t  *in_ptr = base_ptr;
	const uint8_t  *in_ptr_end = in_ptr + line_length;
	uint8_t *base_ptr_out =  PLANE_RAW_DATA(frame_out,0);

	if (flip_x_) {
		uint8_t *out_ptr_end = base_ptr_out+line_length;
		for (yuri::size_t line = 0;line < h; ++line) {
			flip_dispatch(yuv_y_pos, bpp, in_ptr, in_ptr_end, out_ptr_end);
			in_ptr += src_line_advance;
			in_ptr_end += src_line_advance;
			out_ptr_end += line_length;
		}
	} else if (flip_y_) {
		uint8_t *out_ptr = base_ptr_out;
		for (yuri::size_t line = 0;line < h; ++line) {
			std::copy(in_ptr, in_ptr_end, out_ptr);
			in_ptr += src_line_advance;
			in_ptr_end += src_line_advance;
			out_ptr += line_length;
		}
	}
	return frame_out;
}

bool Flip::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name()== "flip_x") {
		flip_x_=parameter.get<bool>();
	} else if (parameter.get_name()== "flip_y") {
		flip_y_=parameter.get<bool>();
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(parameter);
	return true;
}

}

}
