/*!
 * @file 		ColorKey.cpp
 * @author 		<Your name>
 * @date		27.05.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "ColorKey.h"
#include "yuri/core/Module.h"
#include <boost/assign.hpp>
namespace yuri {
namespace color_key {

REGISTER("color_key",ColorKey)

IO_THREAD_GENERATOR(ColorKey)

core::pParameters ColorKey::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("ColorKey");
	p->set_max_pipes(1,1);
	(*p)["r"]["R value of the key color"]=0;
	(*p)["g"]["G value of the key color"]=0;
	(*p)["b"]["B value of the key color"]=0;
	(*p)["delta"]["Threshold for determining same colors"]=90;
	(*p)["delta2"]["Threshold for determining similar colors"]=30;
	(*p)["diff"]["Method for computing differences (linear, quadratic)"]="linear";
	return p;
}


ColorKey::ColorKey(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("color_key")),r_(140),g_(200),b_(75),
delta_(100),delta2_(30),diff_type_(linear)
{
	IO_THREAD_INIT("color_key")
}

ColorKey::~ColorKey()
{
}
namespace {

std::map<std::string, diff_types_> diff_type_strings = boost::assign::map_list_of<std::string, diff_types_>
("linear", 		linear)
("quadratic",	quadratic);

inline ubyte_t diff(ubyte_t a, ubyte_t b)
{
	return a>b?a-b:b-a;
}
inline bool same_color(plane_t::const_iterator data, ubyte_t r, ubyte_t g, ubyte_t b, ushort_t delta)
{
	if (diff(*data++,r)>delta) return false;
	if (diff(*data++,g)>delta) return false;
	if (diff(*data++,b)>delta) return false;
	return true;
}
struct diff_linear {
	static ushort_t combine(ubyte_t a, ubyte_t b, ubyte_t c) {
		return a + b + c;
	}
};
struct diff_quad {
	static ushort_t combine(ubyte_t a, ubyte_t b, ubyte_t c) {
		return a * a + b * b + c * c;
	}
};
template<format_t, class diff_method>
struct simple_kernel {};

template<class diff_method>
struct simple_kernel<YURI_FMT_RGB, diff_method> {
static void eval(plane_t::const_iterator& src_pix, plane_t::iterator& dest_pix, ushort_t total, ushort_t delta, ushort_t delta2)
{
	if (total < delta) {
		*dest_pix++=255;
		*dest_pix++=255;
		*dest_pix++=255;
		*dest_pix++=0;
		src_pix+=3;
	} else if (total < (delta + delta2)) {
		const double a = static_cast<double>(total - delta)/static_cast<double>(delta2);
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=static_cast<ubyte_t>(255*a);
	} else {
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=255;
	}
}
static ushort_t difference(plane_t::const_iterator data, ubyte_t r, ubyte_t g, ubyte_t b)
{
	return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
}
};
template<class diff_method>
struct simple_kernel<YURI_FMT_RGBA, diff_method> {
static void eval(plane_t::const_iterator& src_pix, plane_t::iterator& dest_pix, ushort_t total, ushort_t delta, ushort_t delta2)
{
	if (total < delta) {
		*dest_pix++=255;
		*dest_pix++=255;
		*dest_pix++=255;
		*dest_pix++=0;
		src_pix+=4;
	} else if (total < (delta + delta2)) {
		const double a = static_cast<double>(total - delta)/static_cast<double>(delta2);
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=static_cast<ubyte_t>(*src_pix++*a);
	} else {
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
	}
}
static ushort_t difference(plane_t::const_iterator data, ubyte_t r, ubyte_t g, ubyte_t b)
{
	return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
}
};
template<class diff_method>
struct simple_kernel<YURI_FMT_BGR, diff_method>:
public simple_kernel<YURI_FMT_RGB, diff_method>{

static ushort_t difference(plane_t::const_iterator data, ubyte_t r, ubyte_t g, ubyte_t b)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};
template<class diff_method>
struct simple_kernel<YURI_FMT_BGRA, diff_method>:
public simple_kernel<YURI_FMT_RGBA, diff_method>{

static ushort_t difference(plane_t::const_iterator data, ubyte_t r, ubyte_t g, ubyte_t b)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};

}
template<class kernel>
core::pBasicFrame ColorKey::find_key(const core::pBasicFrame& frame)
{
	const size_t 			width 		= frame->get_width();
	const size_t 			height 		= frame->get_height();
	const size_t 			linesize_in = width * 3;
	const size_t 			linesize_out= width * 4;
	core::pBasicFrame 		outframe 	= allocate_empty_frame(YURI_FMT_RGBA, width, height);
	const plane_t::const_iterator src 		= PLANE_DATA(frame,0).begin();
	const plane_t::iterator 		dest 		= PLANE_DATA(outframe,0).begin();
	for (size_t line = 0; line < height; ++line) {
		plane_t::const_iterator src_pix 	= src+line*linesize_in;
		plane_t::iterator 		dest_pix	= dest+line*linesize_out;
		for (size_t pixel = 0; pixel < width; ++pixel) {
			const ushort_t total = kernel::difference(src_pix, r_, g_, b_);
			kernel::eval(src_pix, dest_pix, total, delta_, delta2_);
		}
	}
	return outframe;
}

template<format_t format>
core::pBasicFrame ColorKey::dispatch_find_key(const core::pBasicFrame& frame)
{
	switch (diff_type_) {
		case linear:
			return find_key<simple_kernel<format, diff_linear> >(frame);
		case quadratic:
			return find_key<simple_kernel<format, diff_quad> >(frame);
	}
	return core::pBasicFrame();
}
bool ColorKey::step()
{
	core::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		const format_t fmt = frame->get_format();
		core::pBasicFrame outframe;
		switch (fmt) {
			case YURI_FMT_RGB:
				outframe = dispatch_find_key<YURI_FMT_RGB>(frame);
				break;
			case YURI_FMT_BGR:
				outframe = dispatch_find_key<YURI_FMT_BGR>(frame);
				break;
			case YURI_FMT_RGBA:
				outframe = dispatch_find_key<YURI_FMT_RGBA>(frame);
				break;
			case YURI_FMT_BGRA:
				outframe = dispatch_find_key<YURI_FMT_BGRA>(frame);
				break;
			default:
				log[log::warning] << "Unsupported frame format";
				return true;
		}

		if (outframe) push_raw_video_frame(0, outframe);
	}
	return true;
}
bool ColorKey::set_param(const core::Parameter& param)
{
//	using boost::iequals;
	if (iequals(param.name,"r")) {
		r_ = param.get<ushort_t>();
	} else if (iequals(param.name,"g")) {
		g_ = param.get<ushort_t>();
	} else if (iequals(param.name,"b")) {
		b_ = param.get<ushort_t>();
	} else if (iequals(param.name,"delta")) {
		delta_ = param.get<ushort_t>();
	} else if (iequals(param.name,"delta2")) {
		delta2_ = param.get<ushort_t>();
	} else if (iequals(param.name,"diff")) {
		const std::string dt = param.get<std::string>();
		if (diff_type_strings.count(dt)) {
			diff_type_ = diff_type_strings[dt];
		} else diff_type_ = linear;
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace color_key */
} /* namespace yuri */

