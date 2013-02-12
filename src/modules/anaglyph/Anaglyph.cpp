/*
 * Anaglyph.cpp
 *
 *  Created on: Jul 31, 2009
 *      Author: neneko
 */

#include "Anaglyph.h"

namespace yuri {

namespace io {

REGISTER("anaglyph",Anaglyph)

shared_ptr<BasicIOThread> Anaglyph::generate(Log &_log,pThreadBase parent,Parameters& parameters)
			throw (Exception)
{
	shared_ptr<Anaglyph> a (new Anaglyph(_log,parent,
			parameters["correction"].get<int>()));
	return a;
}
shared_ptr<Parameters> Anaglyph::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	(*p)["correction"]["Number of pixels the images get shifted (width of the final image gets shorter by the same amount)"]=0;
	(*p)["fast"]["Skips old frames in the pipe. Setting this to false forces processing of all frames"]=true;
	p->set_max_pipes(2,1);
	p->add_input_format(YURI_FMT_RGB);
	p->add_input_format(YURI_FMT_RGBA);
	p->add_output_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGBA);
	return p;
}


Anaglyph::Anaglyph(Log &_log, pThreadBase parent, int correction, bool fast)
	:BasicIOThread(_log,parent,2,1,"Anaglyph"),correction(correction),fast(fast)
{
}

Anaglyph::~Anaglyph() {

}

bool Anaglyph::step()
{
	if (!in[0] || !in[1] || in[0]->is_empty() || in[1]->is_empty())
		return true;
	shared_ptr<BasicFrame> left, right, out_frame;
	if (fast) {
		while (!in[0]->is_empty())	left = in[0]->pop_frame();
		while (!in[1]->is_empty())	right = in[1]->pop_frame();
	} else {
		left = in[0]->pop_frame();
		right = in[1]->pop_frame();
	}
	assert(left && right);

	if (left->get_format() != right->get_format()) {
		log[error] << "Both eyes have different colorspaces, this is not supported now" <<endl;
		return true;
	}
	if (left->get_format() != YURI_FMT_RGB && left->get_format() != YURI_FMT_RGBA) {
		log[error] << "Unsupported color space " << BasicPipe::get_format_string(left->get_format()) << endl;
		return true;
	}
	if (left->get_width() != right->get_width() ||
			left->get_height() != right->get_height()) {
		log[error] << "Images have different resolutions. No support for this now." << endl;
		return true;
	}
	switch (left->get_format()) {
		case YURI_FMT_RGB: {
			assert(sizeof(struct _rgb)==3);
			assert((*left)[0].size == (left->get_width()*left->get_height()*3));
			out_frame = makeAnaglyph<struct _rgb>(left,right);
		} break;
		case YURI_FMT_RGBA: {
			assert(sizeof(struct _rgba)==4);
			assert((*left)[0].size == (left->get_width()*left->get_height()*4));
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

template<typename T> shared_ptr<BasicFrame> Anaglyph::makeAnaglyph(shared_ptr<BasicFrame> left, shared_ptr<BasicFrame> right)
{
	T *datal = (T*) ((*left)[0].data.get());
	T *datar = (T*) ((*right)[0].data.get());
	yuri::size_t w = left->get_width();
	yuri::size_t h = left->get_height();
	shared_ptr<BasicFrame> out_frame;
	if (!correction) { // Simple case with no correction
		out_frame=left;
		for (yuri::size_t i = 0; i < w * h; ++i, ++datal, ++datar) {
			datal->g = datar->g;
			datal->b = datar->b;
		}
	} else if (correction > 0) { // Ok, we have to process correction
		yuri::size_t w_cor = w - correction;
		if (w_cor <= 0) {
			log[error] << "requested corrections larger than image width!"
						<< endl;
			return out_frame;
		}
		yuri::size_t w_rounded = w_cor%4?4*(w_cor/4+1):w_cor;
		out_frame = allocate_empty_frame(left->get_format(),w_rounded,h);
		T *datao = (T *) ((*out_frame)[0].data.get());

		for (yuri::size_t i = 0; i < h; ++i) {
			datal = (T*) (((*left)[0].data.get())) + (w * i);
			datar = (T*) (((*right)[0].data.get())) + (w * i + correction);
			datao = (T*) ((*out_frame)[0].data.get()) + w_rounded*i;
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
			log[error] << "requested corrections larger than image width!"
						<< endl;
			return out_frame;
		}
		yuri::size_t w_rounded = w_cor%4?4*(w_cor/4+1):w_cor;
		out_frame = allocate_empty_frame(left->get_format(),w_rounded,h);
		T *datao = (T *) ((*out_frame)[0].data.get());
		for (yuri::size_t i = 0; i < h; ++i) {
			datal = (T*) (((*left)[0].data.get())) + (w * i - correction);
			datar = (T*) (((*right)[0].data.get())) + (w * i);
			datao = (T*) ((*out_frame)[0].data.get()) + w_rounded*i;
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

