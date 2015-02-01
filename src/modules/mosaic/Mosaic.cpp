/*!
 * @file 		Mosaic.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		02.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Mosaic.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/utils/assign_events.h"
#include <cmath>
namespace yuri {
namespace mosaic {


IOTHREAD_GENERATOR(Mosaic)

MODULE_REGISTRATION_BEGIN("mosaic")
		REGISTER_IOTHREAD("mosaic",Mosaic)
MODULE_REGISTRATION_END()

core::Parameters Mosaic::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("Mosaic");
	p["center"]["Center of mosaic"]=coordinates_t{128,128};
	p["radius"]["Radius of mosaic"]=128;
	p["tile_size"]["Size of a single tile in the mosaic"]=16;
	return p;
}

namespace {
inline position_t get_distance(coordinates_t a, coordinates_t b) {
	position_t x = a.x - b.x;
	position_t y = a.y - b.y;
	return static_cast<position_t>(std::sqrt(x*x + y*y));
}
template<size_t size, class dist_func>
void apply_mosaic(const uint8_t* data_in, uint8_t* data_out, size_t linesize, const coordinates_t& dest_lu, const coordinates_t& dest_rb, dist_func dist)
{
	size_t vals[size];
	std::fill(vals, vals+size, 0);
	size_t count = 0;
	for (position_t line = dest_lu.y; line < dest_rb.y; ++line) {
		const uint8_t* d = data_in + line*linesize + dest_lu.x* size;
		for (position_t col = dest_lu.x; col < dest_rb.x; ++col) {
			for (size_t i = 0; i<size;++i) {
				vals[i] += *d++;
			}
			count ++;
		}
	}

	uint8_t vals2[size];
	for (size_t i = 0; i<size;++i) {
		vals2[i] = count?static_cast<uint8_t>(vals[i] / count):0;
	}
	for (position_t line = dest_lu.y; line < dest_rb.y; ++line) {
		uint8_t* d = data_out + line*linesize + dest_lu.x * size;
		for (position_t col = dest_lu.x; col < dest_rb.x; ++col) {
			if (dist({col, line})) {
				d+=size;
			} else {
				for (size_t i = 0; i<size;++i) {
					*d++ = vals2[i];
				}
			}
		}
	}
}
void process_mosaic(const uint8_t* data_in, uint8_t* data_out, size_t linesize, size_t bpp, coordinates_t img, coordinates_t center, position_t radius, position_t tile_size)
{
	//	log[log::info] << "Number of tiles " << tile_count;

	coordinates_t lu_corner = center - coordinates_t{radius, radius};
	size_t tile_count = tile_size?(2*radius / tile_size):0;
	for (size_t x = 0; x < tile_count+1; ++x) {
		for (size_t y = 0; y < tile_count+1; ++y) {
			coordinates_t corner = lu_corner + coordinates_t{static_cast<position_t>(x*tile_size), static_cast<position_t>(y*tile_size)};

			if (corner.x > img.x) continue;
			if (corner.y > img.y) continue;
			if (corner.x + tile_size < 0) continue;
			if (corner.y + tile_size < 0) continue;
			coordinates_t dest_lu {std::max<position_t>(corner.x, 0L), std::max<position_t>(corner.y, 0L)};
			coordinates_t dest_rb {std::min<position_t>(corner.x+ tile_size, img.x), std::min<position_t>(corner.y + tile_size, img.y)};

			switch (bpp) {
				case 1:	apply_mosaic<1>(data_in, data_out, linesize, dest_lu, dest_rb, [&](const coordinates_t& c)
							{return get_distance(center, c) > radius;}); break;
				case 2:	apply_mosaic<2>(data_in, data_out, linesize, dest_lu, dest_rb, [&](const coordinates_t& c)
							{return get_distance(center, c) > radius;}); break;
				case 3:	apply_mosaic<3>(data_in, data_out, linesize, dest_lu, dest_rb, [&](const coordinates_t& c)
							{return get_distance(center, c) > radius;}); break;
				case 4:	apply_mosaic<4>(data_in, data_out, linesize, dest_lu, dest_rb, [&](const coordinates_t& c)
							{return get_distance(center, c) > radius;}); break;
			}

		}
	}
}

mosaic_detail_t parse_mosaic_info(const event::EventVector& event)
{
	if (event.size() < 2) throw std::runtime_error("Wrong vector size");
	if (event.size() == 3) {
		return mosaic_detail_t {
			event::lex_cast_value<position_t>(event[2]),
			16,
			{event::lex_cast_value<position_t>(event[0]),
			event::lex_cast_value<position_t>(event[1])},
		};
	}
	return mosaic_detail_t {
		event::lex_cast_value<position_t>(event[2]),
		event::lex_cast_value<position_t>(event[3]),
		{event::lex_cast_value<position_t>(event[0]),
		event::lex_cast_value<position_t>(event[1])},
	};
}

using namespace core::raw_format;
const std::vector<format_t> supported_formats = {
		rgb24, bgr24, rgba32, bgra32, argb32, abgr32,
//		yuyv422, uyvy422, yvyu422, vyuy422,
		yuv444,
		y8, u8, v8, depth8, r8, g8, b8
};
}

Mosaic::Mosaic(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("mosaic")),
BasicEventConsumer(log),
mosaics_{{300,50,{100,100}}}
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(supported_formats);
}

Mosaic::~Mosaic() noexcept
{
}

core::pFrame Mosaic::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
	using namespace core::raw_format;

	resolution_t image_size = frame->get_resolution();
	coordinates_t img = {static_cast<position_t>(image_size.width), static_cast<position_t>(image_size.height)};

//	resolution_t rb_corner = center_ + resolution_t{radius_, radius_};

	core::pRawVideoFrame frame_out = dynamic_pointer_cast<core::RawVideoFrame>(frame->get_unique());//core::RawVideoFrame::create_empty(frame->get_format(), image_size);
//	size_t tiles = 2*radius_ / tile_size_;

	const uint8_t * data_in = PLANE_RAW_DATA(frame,0);
	uint8_t * data_out = PLANE_RAW_DATA(frame_out,0);
	size_t linesize = PLANE_DATA(frame,0).get_line_size();

//	const auto& fi = core::raw_format::get_format_info(frame->get_format());
	size_t bpp = core::raw_format::get_fmt_bpp(frame->get_format(),0)/8;
	log[log::verbose_debug] << "Mosaicing " << core::raw_format::get_format_name(frame->get_format());
	for (const auto& x: mosaics_) {
		process_mosaic(data_in, data_out, linesize, bpp, img, x.center, x.radius, x.tile_size);
	}



	return frame_out;
}

bool Mosaic::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(mosaics_[0].center, "center")
			(mosaics_[0].radius, "radius")
			(mosaics_[0].tile_size, "tile_size"))
		return true;
	return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
}
bool Mosaic::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			(mosaics_[0].center.x, "x")
			(mosaics_[0].center.y, "y")
			(mosaics_[0].radius, "radius")
			(mosaics_[0].tile_size, "tile_size")
			(mosaics_[0].center, "center"))
		return true;
	if (event_name == "mosaic" && event->get_type() == event::event_type_t::vector_event) {
		if (auto mosaic_event = dynamic_pointer_cast<event::EventVector>(event)) {
			mosaics_={parse_mosaic_info(*mosaic_event)};
			return true;
		}
	} else if (event_name == "mosaics" && event->get_type() == event::event_type_t::vector_event) {
		if (auto mosaic_events = dynamic_pointer_cast<event::EventVector>(event)) {
			mosaics_.clear();
			log[log::debug] << "Received " << mosaic_events->size() << " mosaics.";
			for (const auto& ev: *mosaic_events) {
				if (ev->get_type() == event::event_type_t::vector_event) {
					if (auto mos_vec = dynamic_pointer_cast<event::EventVector>(ev)) {
						mosaics_.push_back(parse_mosaic_info(*mos_vec));
					}
				}
			}
			return true;
		}
	}
	return true;
}
} /* namespace mosaic */
} /* namespace yuri */
