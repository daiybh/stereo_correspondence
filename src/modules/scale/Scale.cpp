/*!
 * @file 		Scale.cpp
 * @author 		<Your name>
 * @date		09.04.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Scale.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri {
namespace scale {


IOTHREAD_GENERATOR(Scale)

MODULE_REGISTRATION_BEGIN("scale")
		REGISTER_IOTHREAD("scale",Scale)
MODULE_REGISTRATION_END()

core::Parameters Scale::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("Scale");
	p["resolution"]["Resolution to scale to"]=resolution_t{800,600};
	return p;
}


Scale::Scale(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("scale")),
resolution_{800,600}
{
	IOTHREAD_INIT(parameters)
	using namespace core::raw_format;
	set_supported_formats({rgb24, bgr24, rgba32, argb32, bgra32, abgr32, yuv444,
							yuyv422,yvyu422});
//	set_latency(1_ms);
}

Scale::~Scale() noexcept
{
}


namespace {

template<size_t pixel_size>
struct scale_line_bilinear {
	inline static void eval(uint8_t* it, const uint8_t* top, const uint8_t* bottom, const dimension_t new_width, const dimension_t old_width, const double unscale_x, const double y_ratio) {
		const double y_ratio2 = 1.0 - y_ratio;
		for (dimension_t pixel = 0; pixel < new_width -1; ++pixel) {
			const dimension_t left = pixel*unscale_x;
			const dimension_t right = left + 1;
			const double x_ratio = pixel*unscale_x - left;
			const double x_ratio2 = 1.0 - x_ratio;
			for (size_t i = 0; i<pixel_size;++i) {
				*it++ = static_cast<uint8_t>(
					top[left * pixel_size + i] * x_ratio2 * y_ratio2 +
					top[right * pixel_size + i] * x_ratio * y_ratio2 +
					bottom[left * pixel_size + i] * x_ratio2 * y_ratio +
					bottom[right * pixel_size + i] * x_ratio * y_ratio);
			}
		}
		for (size_t i = 0; i<pixel_size;++i) {
			*it++ = static_cast<uint8_t>(
				top[(old_width - 1) * pixel_size + i] * y_ratio2 +
				bottom[(old_width - 1) * pixel_size + i] * y_ratio);
		}
	}
};


inline uint8_t get_y (const dimension_t pixel, const double unscale_x, const uint8_t* top, const uint8_t* bottom, const double y_ratio, const double y_ratio2)
{
	const dimension_t left = pixel*unscale_x;
	const dimension_t right = left + 1;

	const double x_ratio = pixel*unscale_x - left;
	const double x_ratio2 = 1.0 - x_ratio;
	return static_cast<uint8_t>(
		top[left * 2 + 0] * x_ratio2 * y_ratio2 +
		top[right * 2 + 0] * x_ratio * y_ratio2 +
		bottom[left * 2 + 0] * x_ratio2 * y_ratio +
		bottom[right * 2 + 0] * x_ratio * y_ratio);
}
template<size_t adjust>
inline uint8_t get_uv (const dimension_t pixel, const double unscale_x, const uint8_t* top, const uint8_t* bottom, const double y_ratio, const double y_ratio2)
{
	const dimension_t left0 = pixel*unscale_x;
	const dimension_t left = (left0&~1) + adjust;
	const dimension_t right = left + 2;


	const double x_ratio = (pixel*unscale_x - left0 + (left0&1))/2.0;
	const double x_ratio2 = 1.0 - x_ratio;
	return static_cast<uint8_t>(
		top[left * 2 + 1] * x_ratio2 * y_ratio2 +
		top[right * 2 + 1] * x_ratio * y_ratio2 +
		bottom[left * 2 + 1] * x_ratio2 * y_ratio +
		bottom[right * 2 + 1] * x_ratio * y_ratio);
}

struct scale_line_bilinear_yuyv {
	inline static void eval(uint8_t* it, const uint8_t* top, const uint8_t* bottom, const dimension_t new_width, const dimension_t old_width, const double unscale_x, const double y_ratio) {
		const double y_ratio2 = 1.0 - y_ratio;
		for (dimension_t pixel = 0; pixel < new_width -2; pixel+=2) {
			*it++=get_y(pixel, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_uv<0>(pixel, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_y(pixel+1, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_uv<1>(pixel+1, unscale_x, top, bottom, y_ratio, y_ratio2);
		}
		*it++=get_y((old_width - 2), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_uv<0>((old_width - 2), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_y((old_width - 1), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_uv<1>((old_width - 1), unscale_x, top, bottom, y_ratio, y_ratio2);
	}
};
struct scale_line_bilinear_uyvy {
	inline static void eval(uint8_t* it, const uint8_t* top, const uint8_t* bottom, const dimension_t new_width, const dimension_t old_width, const double unscale_x, const double y_ratio) {
		const double y_ratio2 = 1.0 - y_ratio;
		for (dimension_t pixel = 0; pixel < new_width -2; pixel+=2) {
			*it++=get_uv<0>(pixel, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_y(pixel, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_uv<1>(pixel+1, unscale_x, top, bottom, y_ratio, y_ratio2);
			*it++=get_y(pixel+1, unscale_x, top, bottom, y_ratio, y_ratio2);
		}
		*it++=get_uv<0>((old_width - 2), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_y((old_width - 2), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_uv<1>((old_width - 1), unscale_x, top, bottom, y_ratio, y_ratio2);
		*it++=get_y((old_width - 1), unscale_x, top, bottom, y_ratio, y_ratio2);
	}
};
template<class kernel>
core::pRawVideoFrame scale_image(const core::pRawVideoFrame& frame, const resolution_t new_resolution)
{
	auto outframe = core::RawVideoFrame::create_empty(frame->get_format(), new_resolution);
	const auto res = frame->get_resolution();
	const double unscale_x = static_cast<double>(res.width-1)/(new_resolution.width-1);
	const double unscale_y = static_cast<double>(res.height-1)/(new_resolution.height-1);
	const auto linesize_in = PLANE_DATA(frame,0).get_line_size();
	const auto linesize_out = PLANE_DATA(outframe,0).get_line_size();
	const uint8_t* it_in = PLANE_RAW_DATA(frame,0);
	uint8_t* it = PLANE_RAW_DATA(outframe,0);

	for (dimension_t line = 0; line < new_resolution.height -1 ; ++line) {
		const dimension_t top = line*unscale_y;
		const dimension_t bottom = top+1;
		const double y_ratio = line*unscale_y - top;
		kernel::eval(it,
				it_in + top * linesize_in,
				it_in + bottom * linesize_in,
				new_resolution.width,
				res.width,
				unscale_x,
				y_ratio);

		it+=linesize_out;
	}
	kernel::eval(PLANE_RAW_DATA(outframe,0) + (new_resolution.height -1) * linesize_out,
			PLANE_RAW_DATA(frame,0) + (res.height -1) * linesize_in,
			PLANE_RAW_DATA(frame,0) + (res.height -1) * linesize_in,
			new_resolution.width,
			res.width,
			unscale_x,
			0.0);
	return outframe;
}

}

core::pFrame Scale::do_special_single_step(const core::pRawVideoFrame& frame)
{
	using namespace core::raw_format;
	switch (frame->get_format()) {
		case rgb24:
		case bgr24:
		case yuv444:
			return scale_image<scale_line_bilinear<3>>(frame, resolution_);

		case rgba32:
		case argb32:
		case bgra32:
		case abgr32:
			return scale_image<scale_line_bilinear<4>>(frame, resolution_);
		case yuyv422:
		case yvyu422:
			return scale_image<scale_line_bilinear_yuyv>(frame, resolution_);
		case uyvy422:
		case vyuy422:
			return scale_image<scale_line_bilinear_uyvy>(frame, resolution_);
	}
	return {};


}
bool Scale::set_param(const core::Parameter& param)
{
	if (param.get_name() == "resolution") {
		resolution_ = param.get<resolution_t>();
	} else return base_type::set_param(param);
	return true;
}

} /* namespace scale */
} /* namespace yuri */
