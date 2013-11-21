/*!
 * @file 		ColorKey.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		27.05.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "ColorKey.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri {
namespace color_key {

IOTHREAD_GENERATOR(ColorKey)

MODULE_REGISTRATION_BEGIN("color_key")
		REGISTER_IOTHREAD("color_key",ColorKey)
MODULE_REGISTRATION_END()

core::Parameters ColorKey::configure()
{
	core::Parameters p = core::IOFilter::configure();
	p.set_description("ColorKey");
//	p->set_max_pipes(1,1);
	p["r"]["R value of the key color"]=0;
	p["g"]["G value of the key color"]=0;
	p["b"]["B value of the key color"]=0;
	p["delta"]["Threshold for determining same colors"]=90;
	p["delta2"]["Threshold for determining similar colors"]=30;
	p["diff"]["Method for computing differences (linear, quadratic)"]="linear";
	return p;
}


ColorKey::ColorKey(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("color_key")),r_(140),g_(200),b_(75),
delta_(100),delta2_(30),diff_type_(linear)
{
	IOTHREAD_INIT(parameters)
}

ColorKey::~ColorKey() noexcept
{
}
namespace {

std::map<std::string, diff_types_> diff_type_strings = map_list_of<std::string, diff_types_>
("linear", 		linear)
("quadratic",	quadratic);

inline uint8_t diff(uint8_t a, uint8_t b)
{
	return a>b?a-b:b-a;
}
inline bool same_color(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, ssize_t delta)
{
	if (diff(*data++,r)>delta) return false;
	if (diff(*data++,g)>delta) return false;
	if (diff(*data++,b)>delta) return false;
	return true;
}
struct diff_linear {
	static ssize_t combine(uint8_t a, uint8_t b, uint8_t c) {
		return a + b + c;
	}
};
struct diff_quad {
	static ssize_t combine(uint8_t a, uint8_t b, uint8_t c) {
		return a * a + b * b + c * c;
	}
};
template<format_t, class diff_method>
struct simple_kernel {};

template<class diff_method>
struct simple_kernel<core::raw_format::rgb24, diff_method> {
static void eval(core::RawVideoFrame::value_type::const_iterator& src_pix, core::RawVideoFrame::value_type::iterator& dest_pix, ssize_t total, ssize_t delta, ssize_t delta2)
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
		*dest_pix++=static_cast<uint8_t>(255*a);
	} else {
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=255;
	}
}
static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b)
{
	return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
}
};
template<class diff_method>
struct simple_kernel<core::raw_format::rgba32, diff_method> {
static void eval(core::RawVideoFrame::value_type::const_iterator& src_pix, core::RawVideoFrame::value_type::iterator& dest_pix, ssize_t total, ssize_t delta, ssize_t delta2)
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
		*dest_pix++=static_cast<uint8_t>(*src_pix++*a);
	} else {
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
		*dest_pix++=*src_pix++;
	}
}
static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b)
{
	return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
}
};
template<class diff_method>
struct simple_kernel<core::raw_format::bgr24, diff_method>:
public simple_kernel<core::raw_format::rgb24, diff_method>{

static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};
template<class diff_method>
struct simple_kernel<core::raw_format::abgr32, diff_method>:
public simple_kernel<core::raw_format::rgba32, diff_method>{

static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};

}
template<class kernel>
core::pRawVideoFrame ColorKey::find_key(const core::pRawVideoFrame& frame)
{
	const resolution_t		res 		= frame->get_resolution();
	const size_t 			width 		= res.width;
	const size_t 			height 		= res.height;
	const size_t 			linesize_in = width * 3;
	const size_t 			linesize_out= width * 4;
	core::pRawVideoFrame	outframe 	= core::RawVideoFrame::create_empty(core::raw_format::rgba32, res);
	const core::RawVideoFrame::value_type::const_iterator src 		= PLANE_DATA(frame,0).begin();
	const core::RawVideoFrame::value_type::iterator 	dest 		= PLANE_DATA(outframe,0).begin();
	for (size_t line = 0; line < height; ++line) {
		core::RawVideoFrame::value_type::const_iterator src_pix 	= src+line*linesize_in;
		core::RawVideoFrame::value_type::iterator 		dest_pix	= dest+line*linesize_out;
		for (size_t pixel = 0; pixel < width; ++pixel) {
			const ssize_t total = kernel::difference(src_pix, r_, g_, b_);
			kernel::eval(src_pix, dest_pix, total, delta_, delta2_);
		}
	}
	return outframe;
}

template<format_t format>
core::pRawVideoFrame ColorKey::dispatch_find_key(const core::pRawVideoFrame& frame)
{
	switch (diff_type_) {
		case linear:
			return find_key<simple_kernel<format, diff_linear> >(frame);
		case quadratic:
			return find_key<simple_kernel<format, diff_quad> >(frame);
	}
	return core::pRawVideoFrame();
}
//bool ColorKey::step()
core::pFrame ColorKey::do_special_single_step(const core::pRawVideoFrame& frame)
{
//	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(gframe);
//	if (frame) {
	const format_t fmt = frame->get_format();
	core::pRawVideoFrame outframe;
	switch (fmt) {
		case core::raw_format::rgb24:
			outframe = dispatch_find_key<core::raw_format::rgb24>(frame);
			break;
		case core::raw_format::bgr24:
			outframe = dispatch_find_key<core::raw_format::bgr24>(frame);
			break;
		case core::raw_format::rgba32:
			outframe = dispatch_find_key<core::raw_format::rgba32>(frame);
			break;
		case core::raw_format::abgr32:
			outframe = dispatch_find_key<core::raw_format::abgr32>(frame);
			break;
		default:
			log[log::warning] << "Unsupported frame format";
			return {};
	}

	return outframe;
//	if (outframe) push_frame(0, outframe);
//	}
//	return {};
}
bool ColorKey::set_param(const core::Parameter& param)
{
//	using boost::iequals;
	if (iequals(param.get_name(),"r")) {
		r_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"g")) {
		g_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"b")) {
		b_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"delta")) {
		delta_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"delta2")) {
		delta2_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"diff")) {
		const std::string dt = param.get<std::string>();
		if (diff_type_strings.count(dt)) {
			diff_type_ = diff_type_strings[dt];
		} else diff_type_ = linear;
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace color_key */
} /* namespace yuri */

