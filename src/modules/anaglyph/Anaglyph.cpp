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
#include "yuri/core/RegisteredClass.h"
#include "yuri/core/pipe_types.h"
#include "yuri/core/BasicPipe.h"
namespace yuri {

namespace anaglyph {

REGISTER("anaglyph",Anaglyph)

IO_THREAD_GENERATOR(Anaglyph)
core::pParameters Anaglyph::configure()
{
	core::pParameters p = core::BasicMultiIOFilter::configure();
	(*p)["correction"]["Number of pixels the images get shifted (width of the final image gets shorter by the same amount)"]=0;
	//(*p)["fast"]["Skips old frames in the pipe. Setting this to false forces processing of all frames"]=true;
	p->set_max_pipes(2,1);
	p->add_input_format(YURI_FMT_RGB);
	p->add_input_format(YURI_FMT_RGBA);
	p->add_output_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGBA);
	return p;
}


Anaglyph::Anaglyph(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters)
	:core::BasicMultiIOFilter(_log,parent,2,1,"Anaglyph"),correction(0)//,fast(true)
{
	IO_THREAD_INIT("Anaglyph")
}

Anaglyph::~Anaglyph() {

}

std::vector<core::pBasicFrame> Anaglyph::do_single_step(const std::vector<core::pBasicFrame>& frames)
{
	assert(frames.size()==2);
	auto& left = frames[0];
	auto& right = frames[1];
	assert(left && right);
	core::pBasicFrame out_frame;

	if (left->get_format() != right->get_format()) {
		log[log::error] << "The eyes have different colorspaces, this is not supported now\n";
		return {};
	}
	if (left->get_format() != YURI_FMT_RGB && left->get_format() != YURI_FMT_RGBA) {
		log[log::error] << "Unsupported color space " << core::BasicPipe::get_format_string(left->get_format()) << "\n";
		return {};
	}
	if (left->get_width() != right->get_width() ||
			left->get_height() != right->get_height()) {
		log[log::error] << "Images have different resolutions. No support for this now.\n";
		return {};
	}
	switch (left->get_format()) {
		case YURI_FMT_RGB: {
			assert(sizeof(struct _rgb)==3);
			assert(PLANE_SIZE(left,0) == (left->get_width()*left->get_height()*3));
			out_frame = makeAnaglyph<struct _rgb>(left,right);
		} break;
		case YURI_FMT_RGBA: {
			assert(sizeof(struct _rgba)==4);
			assert(PLANE_SIZE(left,0) == (left->get_width()*left->get_height()*4));
			out_frame = makeAnaglyph<struct _rgba>(left, right);
		} break;
	//	default:
			//return false;// Something very fishy is going on!
	}

	if (!out_frame || !out[0]) return {};
	out_frame->set_time(left->get_pts(),left->get_dts(),left->get_duration());
	return {out_frame};
//	push_raw_video_frame(0,out_frame);
//	return true;

}

template<typename T> core::pBasicFrame Anaglyph::makeAnaglyph(const core::pBasicFrame& left, const core::pBasicFrame& right)
{
	const T *datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0));
	const T *datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0));
	yuri::size_t w = left->get_width();
	yuri::size_t h = left->get_height();
	core::pBasicFrame out_frame;
	if (!correction) { // Simple case with no correction
		out_frame=left->get_copy();
		T *datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0));
		for (yuri::size_t i = 0; i < w * h; ++i, ++datal, ++datar) {
			datao->r = datal->r;
			datao->g = datar->g;
			datao->b = datar->b;
		}
	} else if (correction > 0) { // Ok, we have to process correction
		yuri::size_t w_cor = w - correction;
		if (w_cor <= 0) {
			log[log::error] << "requested corrections larger than image width!\n";
			return out_frame;
		}
		yuri::size_t w_rounded = w_cor%4?4*(w_cor/4+1):w_cor;
		out_frame = allocate_empty_frame(left->get_format(),w_rounded,h);
		T *datao ;//= reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0));

		for (yuri::size_t i = 0; i < h; ++i) {
			datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0)) + (w * i);
			datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0)) + (w * i + correction);
			datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0)) + w_rounded*i;
			for (yuri::size_t j = 0; j < w_cor; ++j, ++datal, ++datar, ++datao) {
				datao->r = datal->r;
				datao->b = datar->b;
				datao->g = datar->g;
			}
			for (yuri::size_t j = w_cor; j < w_rounded; ++j, ++datao) {
//				memset(datao,0,sizeof(T));
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
		out_frame = allocate_empty_frame(left->get_format(),w_rounded,h);
		T *datao;// = (T *) ((*out_frame)[0].data.get());
		for (yuri::size_t i = 0; i < h; ++i) {
			datal = reinterpret_cast<T*>(PLANE_RAW_DATA(left,0)) + (w * i - correction);
			datar = reinterpret_cast<T*>(PLANE_RAW_DATA(right,0)) + (w * i);
			datao = reinterpret_cast<T*>(PLANE_RAW_DATA(out_frame,0)) + w_rounded*i;

			for (yuri::size_t j = 0; j < w_cor; ++j, ++datal, ++datar, ++datao) {
				datao->r = datal->r;
				datao->b = datar->b;
				datao->g = datar->g;
			}
			for (yuri::size_t j = w_cor; j < w_rounded; ++j, ++datao) {
				std::fill_n(reinterpret_cast<char*>(datao),sizeof(T),0);
//				memset(datao,0,sizeof(T));
			}
		}
	}
	return out_frame;
}

bool Anaglyph::set_param(const core::Parameter& param)
{
	if (param.name == "correction") {
		correction = param.get<int>();
	} else return BasicMultiIOFilter::set_param(param);
	return true;
}

}

}

