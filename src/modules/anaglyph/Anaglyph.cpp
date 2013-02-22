/*!
 * @file 		Anaglyph.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Anaglyph.h"
#include "yuri/core/RegisteredClass.h"
#include "yuri/core/pipe_types.h"
#include "yuri/core/BasicPipe.h"
namespace yuri {

namespace anaglyph {

REGISTER("anaglyph",Anaglyph)

core::pBasicIOThread Anaglyph::generate(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters)

{
	shared_ptr<Anaglyph> a (new Anaglyph(_log,parent,
			parameters["correction"].get<int>()));
	return a;
}
core::pParameters Anaglyph::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["correction"]["Number of pixels the images get shifted (width of the final image gets shorter by the same amount)"]=0;
	(*p)["fast"]["Skips old frames in the pipe. Setting this to false forces processing of all frames"]=true;
	p->set_max_pipes(2,1);
	p->add_input_format(YURI_FMT_RGB);
	p->add_input_format(YURI_FMT_RGBA);
	p->add_output_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGBA);
	return p;
}


Anaglyph::Anaglyph(log::Log &_log, core::pwThreadBase parent, int correction, bool fast)
	:core::BasicIOThread(_log,parent,2,1,"Anaglyph"),correction(correction),fast(fast)
{
}

Anaglyph::~Anaglyph() {

}

bool Anaglyph::step()
{
	if (!in[0] || !in[1] || in[0]->is_empty() || in[1]->is_empty())
		return true;
	core::pBasicFrame left, right, out_frame;
	if (fast) {
		while (!in[0]->is_empty())	left = in[0]->pop_frame();
		while (!in[1]->is_empty())	right = in[1]->pop_frame();
	} else {
		left = in[0]->pop_frame();
		right = in[1]->pop_frame();
	}
	assert(left && right);

	if (left->get_format() != right->get_format()) {
		log[log::error] << "Both eyes have different colorspaces, this is not supported now\n";
		return true;
	}
	if (left->get_format() != YURI_FMT_RGB && left->get_format() != YURI_FMT_RGBA) {
		log[log::error] << "Unsupported color space " << core::BasicPipe::get_format_string(left->get_format()) << "\n";
		return true;
	}
	if (left->get_width() != right->get_width() ||
			left->get_height() != right->get_height()) {
		log[log::error] << "Images have different resolutions. No support for this now.\n";
		return true;
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

	if (!out_frame || !out[0]) return true;
	out_frame->set_time(left->get_pts(),left->get_dts(),left->get_duration());
	push_raw_video_frame(0,out_frame);
	return true;
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
				memset(datao,0,sizeof(T));
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
				memset(datao,0,sizeof(T));
			}
		}
	}
	return out_frame;
}

}

}

