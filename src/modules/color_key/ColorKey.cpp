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
	core::Parameters p = base_type::configure();
	p.set_description("ColorKey");
	p["r"]["R value of the key color"]=0;
	p["g"]["G value of the key color"]=0;
	p["b"]["B value of the key color"]=0;
	p["y"]["Y value of the key color"]=0;
	p["u"]["U value of the key color"]=0;
	p["v"]["V value of the key color"]=0;
	p["y_cutoff"]["Unimportance of Y value. Set to 1 to keep full y range and to 256 to completely ignore Y value"]=5;
	p["delta"]["Threshold for determining same colors"]=90;
	p["delta2"]["Threshold for determining similar colors"]=30;
	p["diff"]["Method for computing differences (linear, quadratic)"]="linear";
	return p;
}


ColorKey::ColorKey(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("color_key")),
event::BasicEventConsumer(log),
r_(140),g_(200),b_(75),y_(0),u_(128),v_(128),y_cutoff_(5),delta_(100),delta2_(30),
diff_type_(linear)
{
	IOTHREAD_INIT(parameters)
	using namespace core::raw_format;
	set_supported_formats({rgb24, bgr24, rgba32, bgra32, yuyv422, yuv444, yuva4444});
}

ColorKey::~ColorKey() noexcept
{
}
namespace {

std::map<std::string, diff_types_> diff_type_strings = {
		{"linear", 		linear},
		{"quadratic",	quadratic}};

inline uint8_t diff(uint8_t a, uint8_t b)
{
	return a>b?a-b:b-a;
}
//inline bool same_color(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, ssize_t delta)
//{
//	if (diff(*data++,r)>delta) return false;
//	if (diff(*data++,g)>delta) return false;
//	if (diff(*data++,b)>delta) return false;
//	return true;
//}
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
	static constexpr format_t out_type = core::raw_format::rgba32;
	static constexpr bool rgb_vals = true;
	static constexpr dimension_t get_width(dimension_t width) { return width; }
	static constexpr dimension_t get_height(dimension_t height) { return height; }
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
	static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, int)
	{
		return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
	}
};
template<class diff_method>
struct simple_kernel<core::raw_format::rgba32, diff_method> {
	static const format_t out_type = core::raw_format::rgba32;
	static constexpr bool rgb_vals = true;
	static constexpr dimension_t get_width(dimension_t width) { return width; }
	static constexpr dimension_t get_height(dimension_t height) { return height; }
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
	static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, int)
	{
		return diff_method::combine(diff(*(data+0),r), diff(*(data+1),g), diff(*(data+2),b));
	}
};

template<class diff_method>
struct simple_kernel<core::raw_format::bgr24, diff_method>:
public simple_kernel<core::raw_format::rgb24, diff_method>{
	static const format_t out_type = core::raw_format::bgra32;
static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, int)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};
template<class diff_method>
struct simple_kernel<core::raw_format::abgr32, diff_method>:
public simple_kernel<core::raw_format::rgba32, diff_method>{
	static const format_t out_type = core::raw_format::bgra32;
static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t r, uint8_t g, uint8_t b, int)
{
	return diff_method::combine(diff(*(data+0),b), diff(*(data+1),g), diff(*(data+2),r));
}
};

template<class diff_method>
struct simple_kernel<core::raw_format::yuv444, diff_method> {
	static const format_t out_type = core::raw_format::yuva4444;
	static constexpr bool rgb_vals = false;
	static constexpr dimension_t get_width(dimension_t width) { return width; }
	static constexpr dimension_t get_height(dimension_t height) { return height; }
	static void eval(core::RawVideoFrame::value_type::const_iterator& src_pix, core::RawVideoFrame::value_type::iterator& dest_pix, ssize_t total, ssize_t delta, ssize_t delta2)
	{
		if (total < delta) { // transparent
			*dest_pix++=255;
			*dest_pix++=128;
			*dest_pix++=128;
			*dest_pix++=0;
			src_pix+=3;
		} else if (total < (delta + delta2)) {
			const double a = static_cast<double>(total - delta)/static_cast<double>(delta2);
			const uint8_t alpha = static_cast<uint8_t>(255*a);
			*dest_pix++=*src_pix++;
			*dest_pix++=*src_pix++;
			*dest_pix++=*src_pix++;
			*dest_pix++=alpha;
		} else {
			*dest_pix++=*src_pix++;
			*dest_pix++=*src_pix++;
			*dest_pix++=*src_pix++;
			*dest_pix++=255;
		}
	}
	static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t y, uint8_t u, uint8_t v, int y_cutoff)
	{
		return diff_method::combine(diff(*(data+0),y)/y_cutoff, diff(*(data+1),u), diff(*(data+3),v));
	}
};
template<class diff_method>
struct simple_kernel<core::raw_format::yuva4444, diff_method> {
	static const format_t out_type = core::raw_format::yuva4444;
	static constexpr bool rgb_vals = false;
	static constexpr dimension_t get_width(dimension_t width) { return width; }
	static constexpr dimension_t get_height(dimension_t height) { return height; }
	static void eval(core::RawVideoFrame::value_type::const_iterator& src_pix, core::RawVideoFrame::value_type::iterator& dest_pix, ssize_t total, ssize_t delta, ssize_t delta2)
	{
		if (total < delta) { // transparent
			*dest_pix++=255;
			*dest_pix++=128;
			*dest_pix++=128;
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
	static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t y, uint8_t u, uint8_t v, int y_cutoff)
	{
		return diff_method::combine(diff(*(data+0),y)/y_cutoff, diff(*(data+1),u), diff(*(data+3),v));
	}
};
template<class diff_method>
struct simple_kernel<core::raw_format::yuyv422, diff_method> {
	static const format_t out_type = core::raw_format::yuva4444;
	static constexpr bool rgb_vals = false;
	static constexpr dimension_t get_width(dimension_t width) { return width/2; }
	static constexpr dimension_t get_height(dimension_t height) { return height; }
	static void eval(core::RawVideoFrame::value_type::const_iterator& src_pix, core::RawVideoFrame::value_type::iterator& dest_pix, ssize_t total, ssize_t delta, ssize_t delta2)
	{
		if (total < delta) { // transparent
			*dest_pix++=255;
			*dest_pix++=128;
			*dest_pix++=128;
			*dest_pix++=0;
			*dest_pix++=255;
			*dest_pix++=128;
			*dest_pix++=128;
			*dest_pix++=0;
			src_pix+=4;
		} else if (total < (delta + delta2)) {
			const double a = static_cast<double>(total - delta)/static_cast<double>(delta2);
			const uint8_t alpha = static_cast<uint8_t>(255*a);
			*dest_pix++=*src_pix++;
			const uint8_t u = *src_pix++;
			*dest_pix++=u;
			const uint8_t y2 = *src_pix++;
			const uint8_t v = *src_pix++;
			*dest_pix++=v;
			*dest_pix++=alpha;
			*dest_pix++=y2;
			*dest_pix++=u;
			*dest_pix++=v;
			*dest_pix++=alpha;
		} else {
			*dest_pix++=*src_pix++;
			const uint8_t u = *src_pix++;
			*dest_pix++=u;
			const uint8_t y2 = *src_pix++;
			const uint8_t v = *src_pix++;
			*dest_pix++=v;
			*dest_pix++=255;
			*dest_pix++=y2;
			*dest_pix++=u;
			*dest_pix++=v;
			*dest_pix++=255;
		}
	}
	static ssize_t difference(core::RawVideoFrame::value_type::const_iterator data, uint8_t y, uint8_t u, uint8_t v, int y_cutoff)
	{
		return diff_method::combine(diff(*(data+0)/2+*(data+2)/2,y)/y_cutoff, diff(*(data+1),u), diff(*(data+3),v));
	}
};

}
template<class kernel>
core::pRawVideoFrame ColorKey::find_key(const core::pRawVideoFrame& frame)
{
	const resolution_t		res 		= frame->get_resolution();
	const dimension_t		width 		= kernel::get_width(res.width);
	const dimension_t		height 		= kernel::get_height(res.height);
	const format_t			format_out	= kernel::out_type;
	core::pRawVideoFrame	outframe 	= core::RawVideoFrame::create_empty(format_out, res);
	const size_t 			linesize_in = PLANE_DATA(frame,0).get_line_size();
	const size_t 			linesize_out= PLANE_DATA(outframe,0).get_line_size();


	const core::RawVideoFrame::value_type::const_iterator src 		= PLANE_DATA(frame,0).begin();
	const core::RawVideoFrame::value_type::iterator 	dest 		= PLANE_DATA(outframe,0).begin();
	for (size_t line = 0; line < height; ++line) {
		core::RawVideoFrame::value_type::const_iterator src_pix 	= src+line*linesize_in;
		core::RawVideoFrame::value_type::iterator 		dest_pix	= dest+line*linesize_out;
		for (size_t pixel = 0; pixel < width; ++pixel) {
			if (kernel::rgb_vals) { // this condition should get optimized out by compiler...
				const ssize_t total = kernel::difference(src_pix, r_, g_, b_, static_cast<int>(y_cutoff_));
				kernel::eval(src_pix, dest_pix, total, delta_, delta2_);
			} else {
				const ssize_t total = kernel::difference(src_pix, y_, u_, v_, static_cast<int>(y_cutoff_));
				kernel::eval(src_pix, dest_pix, total, delta_, delta2_);
			}

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

core::pFrame ColorKey::do_special_single_step(const core::pRawVideoFrame& frame)
{
	process_events();
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
		case core::raw_format::yuyv422:
			outframe = dispatch_find_key<core::raw_format::yuyv422>(frame);
			break;
		case core::raw_format::yuv444:
			outframe = dispatch_find_key<core::raw_format::yuv444>(frame);
			break;
		case core::raw_format::yuva4444:
			outframe = dispatch_find_key<core::raw_format::yuva4444>(frame);
			break;
		default:
			log[log::warning] << "Unsupported frame format";
			return {};
	}

	return outframe;
}
bool ColorKey::set_param(const core::Parameter& param)
{
	if (iequals(param.get_name(),"r")) {
		r_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"g")) {
		g_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"b")) {
		b_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"y")) {
		y_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"u")) {
		u_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"v")) {
		v_ = param.get<uint8_t>();
	} else if (iequals(param.get_name(),"delta")) {
		delta_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"delta2")) {
		delta2_ = param.get<ssize_t>();
	} else if (iequals(param.get_name(),"y_cutoff")) {
		y_cutoff_ = param.get<size_t>();
		if (y_cutoff_ < 1) y_cutoff_ = 1;
	} else if (iequals(param.get_name(),"diff")) {
		const std::string dt = param.get<std::string>();
		if (diff_type_strings.count(dt)) {
			diff_type_ = diff_type_strings[dt];
		} else diff_type_ = linear;
	} else return base_type::set_param(param);
	return true;
}

bool ColorKey::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (event_name == "yuv") {
		auto val = event::get_value<event::EventVector>(event);
		if (val.size() < 3) return false;
		y_ = event::lex_cast_value<uint8_t>(val[0]);
		u_ = event::lex_cast_value<uint8_t>(val[1]);
		v_ = event::lex_cast_value<uint8_t>(val[2]);
	} else if (event_name == "rgb") {
		auto val = event::get_value<event::EventVector>(event);
		if (val.size() < 3) return false;
		r_ = event::lex_cast_value<uint8_t>(val[0]);
		g_ = event::lex_cast_value<uint8_t>(val[1]);
		b_ = event::lex_cast_value<uint8_t>(val[2]);
	} else if (event_name == "delta") {
		delta_ = event::lex_cast_value<ssize_t>(event);
	} else if (event_name == "delta2") {
		delta2_ = event::lex_cast_value<ssize_t>(event);
	} else if (event_name == "y_cutoff") {
		y_cutoff_ = event::lex_cast_value<size_t>(event);
	}
	return true;
}
} /* namespace color_key */
} /* namespace yuri */

