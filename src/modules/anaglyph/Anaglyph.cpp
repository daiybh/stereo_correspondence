/*!
 * @file 		Anaglyph.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Anaglyph.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri {

namespace anaglyph {


MODULE_REGISTRATION_BEGIN("anaglyph")
	REGISTER_IOTHREAD("anaglyph",Anaglyph)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(Anaglyph)

core::Parameters Anaglyph::configure()
{
	core::Parameters p = anaglyph_base::configure();
	p["correction"]["Number of pixels the images get shifted (width of the final image gets shorter by the same amount)"]=0;
	return p;
}


namespace {
template<typename T>
typename std::enable_if<sizeof(T)==3||sizeof(T)==6,void>::type
copy_data(T* datao, const T* datal, const T* datar)
{
	datao->r = datal->r;
	datao->g = datar->g;
	datao->b = datar->b;
}
template<typename T>
typename std::enable_if<sizeof(T)==4||sizeof(T)==8,void>::type
copy_data(T* datao, const T* datal, const T* datar)
{
	datao->r = datal->r;
	datao->g = datar->g;
	datao->b = datar->b;
	datao->a = datal->a;
}


template<typename T> core::pFrame makeAnaglyph(log::Log& log, const core::pRawVideoFrame& left, const core::pRawVideoFrame& right, const int correction)
{
	assert(PLANE_SIZE(left,0) == (left->get_width()*left->get_height()*sizeof(T)));
	const T *datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0));
	const T *datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0));
	yuri::size_t w = left->get_width();
	yuri::size_t h = left->get_height();
	core::pRawVideoFrame out_frame;
	if (!correction) { // Simple case with no correction
		//out_frame=left->get_copy();
		out_frame = core::RawVideoFrame::create_empty(left->get_format(), left->get_resolution());
		T *datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0));
		for (yuri::size_t i = 0; i < w * h; ++i, ++datal, ++datar) {
			copy_data(datao, datal, datar);
		}
	} else if (correction > 0) { // Ok, we have to process correction
		yuri::size_t w_cor = w - correction;
		if (w_cor <= 0) {
			log[log::error] << "requested corrections larger than image width!\n";
			return out_frame;
		}
		yuri::size_t w_rounded = w_cor%4?4*(w_cor/4+1):w_cor;
		out_frame = core::RawVideoFrame::create_empty(left->get_format(),{w_rounded,h});
		T *datao ;//= reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0));

		for (yuri::size_t i = 0; i < h; ++i) {
			datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0)) + (w * i);
			datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0)) + (w * i + correction);
			datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0)) + w_rounded*i;
			for (yuri::size_t j = 0; j < w_cor; ++j, ++datal, ++datar, ++datao) {
				copy_data(datao, datal, datar);
			}
			for (yuri::size_t j = w_cor; j < w_rounded; ++j, ++datao) {
				std::fill_n(reinterpret_cast<char*>(datao),sizeof(T),0);
			}
		}
	} else { // Ok, we have to process  negative correction
		yuri::size_t w_cor = w + correction;
		if (w_cor <= 0) {
			log[log::error] << "requested corrections larger than image width!\n";
			return out_frame;
		}
		yuri::size_t w_rounded = w_cor%4?4*(w_cor/4+1):w_cor;
		out_frame = core::RawVideoFrame::create_empty(left->get_format(),{w_rounded,h});
		T *datao;// = (T *) ((*out_frame)[0].data.get());
		for (yuri::size_t i = 0; i < h; ++i) {
			datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0)) + (w * i - correction);
			datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0)) + (w * i);
			datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0)) + w_rounded*i;

			for (yuri::size_t j = 0; j < w_cor; ++j, ++datal, ++datar, ++datao) {
				copy_data(datao, datal, datar);
			}
			for (yuri::size_t j = w_cor; j < w_rounded; ++j, ++datao) {
				std::fill_n(reinterpret_cast<char*>(datao),sizeof(T),0);
			}
		}
	}
	return out_frame;
}

PACK_START
struct rgb_t {
	uint8_t r,g,b;
} PACK_END;

PACK_START struct rgba_t {
	uint8_t r,g,b,a;
} PACK_END;

struct rgb48_t {
	uint16_t r,g,b;
} PACK_END;

PACK_START struct rgba64_t {
	uint16_t r,g,b,a;
} PACK_END;



PACK_START struct bgr_t {
	uint8_t b,g,r;
} PACK_END;

PACK_START struct abgr_t {
	uint8_t a,b,g,r;
} PACK_END;

PACK_START struct bgr48_t {
	uint16_t b,g,r;
} PACK_END;

PACK_START struct abgr64_t {
	uint16_t a,b,g,r;
} PACK_END;


PACK_START struct argb_t {
	uint8_t a,r,g,b;
} PACK_END;


PACK_START struct argb64_t {
	uint16_t a,r,g,b;
} PACK_END;

PACK_START struct bgra_t {
	uint8_t a,b,g,r;
} PACK_END;


PACK_START struct bgra64_t {
	uint16_t a,b,g,r;
} PACK_END;


}




Anaglyph::Anaglyph(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters)
	:anaglyph_base(_log,parent,1,"Anaglyph"),correction(0)
{
	IOTHREAD_INIT(parameters)
	// Assert correct packing
	static_assert((sizeof(rgb_t) == 3) && (sizeof(bgr_t) == 3) &&
				(sizeof(rgba_t) == 4) && (sizeof(argb_t) == 4) &&
				(sizeof(bgra_t) == 4) && (sizeof(abgr_t) == 4) &&
				(sizeof(rgb48_t) == 6) && (sizeof(bgr48_t) == 6) &&
				(sizeof(rgba64_t) == 8) && (sizeof(argb64_t) == 8) &&
				(sizeof(bgra64_t) == 8) && (sizeof(abgr64_t) == 8),
				"Wrong packing");
}

Anaglyph::~Anaglyph() noexcept {

}

//std::vector<core::pBasicFrame> Anaglyph::do_single_step(const std::vector<core::pBasicFrame>& frames)
std::vector<core::pFrame>	Anaglyph::do_special_step(const std::tuple<core::pRawVideoFrame, core::pRawVideoFrame>& frames)
{
//	assert(frames.size()==2);
	auto& left = std::get<0>(frames);
	auto& right = std::get<1>(frames);
	assert(left && right);
	core::pFrame out_frame;


	if (left->get_format() != right->get_format()) {
		log[log::error] << "The eyes have different colorspaces, this is not supported no";
		return {};
	}
	if (left->get_format() != core::raw_format::rgb24 && left->get_format() != core::raw_format::rgba32) {
		log[log::error] << "Unsupported color space ";// << core::BasicPipe::get_format_string(left->get_format()) << "\n";
		return {};
	}
	if (left->get_width() != right->get_width() ||
			left->get_height() != right->get_height()) {
		log[log::error] << "Images have different resolutions. No support for this now.\n";
		return {};
	}
	switch (left->get_format()) {
		case core::raw_format::rgb24: {
			out_frame = makeAnaglyph<struct rgb_t>(log, left,right, correction);
		} break;
		case core::raw_format::rgba32: {
			out_frame = makeAnaglyph<struct rgba_t>(log, left, right, correction);
		} break;
		case core::raw_format::bgr24: {
			out_frame = makeAnaglyph<struct bgr_t>(log, left,right, correction);
		} break;
		case core::raw_format::abgr32: {
			out_frame = makeAnaglyph<struct abgr_t>(log, left, right, correction);
		} break;
		case core::raw_format::rgb48: {
			out_frame = makeAnaglyph<struct rgb48_t>(log, left,right, correction);
		} break;
		case core::raw_format::rgba64: {
			out_frame = makeAnaglyph<struct rgba64_t>(log, left, right, correction);
		} break;
		case core::raw_format::bgr48: {
			out_frame = makeAnaglyph<struct bgr48_t>(log, left,right, correction);
		} break;
		case core::raw_format::abgr64: {
			out_frame = makeAnaglyph<struct abgr64_t>(log, left, right, correction);
		} break;
		case core::raw_format::argb32: {
			out_frame = makeAnaglyph<struct argb_t>(log, left, right, correction);
		} break;
		case core::raw_format::argb64: {
			out_frame = makeAnaglyph<struct argb64_t>(log, left, right, correction);
		} break;
		case core::raw_format::bgra32: {
			out_frame = makeAnaglyph<struct bgra_t>(log, left, right, correction);
		} break;
		case core::raw_format::bgra64: {
			out_frame = makeAnaglyph<struct bgra64_t>(log, left, right, correction);
		} break;
	//	default:
			//return false;// Something very fishy is going on!
	}

//	if (!out_frame || !out[0]) return {};
//	out_frame->set_time(left->get_pts(),left->get_dts(),left->get_duration());
	return {out_frame};
//	push_raw_video_frame(0,out_frame);
//	return true;

}


bool Anaglyph::set_param(const core::Parameter& param)
{
	if (param.get_name() == "correction") {
		correction = param.get<int>();
	} else return anaglyph_base::set_param(param);
	return true;
}

}

}

