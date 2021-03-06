/*!
 * @file 		AVScaler.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "AVScaler.h"
#include "yuri/core/Module.h"

namespace yuri
{
namespace video
{

REGISTER("avscaler",AVScaler)


IO_THREAD_GENERATOR(AVScaler)
//core::pIOThread AVScaler::generate(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters)
//{
//	long format = core::BasicPipe::get_format_from_string(parameters["format"].get<std::string>(),YURI_FMT);
//	if (format == YURI_FMT_NONE) throw exception::InitializationFailed(
//		std::string("Wrong output format ")+parameters["format"].get<std::string>());
//	shared_ptr<AVScaler> s(new AVScaler(_log,parent));
//	s->set_output_format(parameters["width"].get<yuri::ssize_t>(),
//			parameters["height"].get<yuri::ssize_t>(),
//			format);
//	return s;
//}
core::pParameters AVScaler::configure()
{
	core::pParameters p =IOThread::configure();
	p->set_description("Scaling object using libswscale");
	(*p)["format"]["Color format for the output"]="RGB";
	(*p)["width"]["Width of the output image. Set to negative value to disable scaling"]=640;
	(*p)["height"]["Height of the output image. Set to negative value to disable scaling"]=480;
	p->set_max_pipes(1,1);
	for(const auto& fmt: get_supported_formats()) {
		p->add_output_format(fmt);
		p->add_input_format(fmt);
		for (const auto& fmt2: get_supported_formats()) {
			p->add_converter(fmt,fmt2,0,true);
		}
	}
	return p;
}

bool AVScaler::configure_converter(core::Parameters& parameters,long format_in,
		long format_out)
{
	std::set<long> fmts = get_supported_formats();
	if (fmts.find(format_in) == fmts.end()) throw exception::NotImplemented();
	if (fmts.find(format_out) == fmts.end()) throw exception::NotImplemented();
	parameters["format"]=core::BasicPipe::get_simple_format_string(format_out);
	parameters["width"]=-1;
	parameters["height"]=-1;
	return true;
}


AVScaler::AVScaler(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters):
	BasicIOFilter(_log, parent, "AVScaler"),AVCodecBase(BasicIOFilter::log),
	f_in(PIX_FMT_NONE),f_out(PIX_FMT_NONE),
	format_in(YURI_FMT_NONE),format_out(YURI_FMT_NONE),w_in(0),h_in(0),w_out(0),
	h_out(0),scaling(false),transforming(false),valid_contexts(false),
	input_pipe_connected(false),scaling_disabled(false),pts(0),duration(0)
{
	latency=100000;
	IO_THREAD_INIT("AVScaler")
	long format = core::BasicPipe::get_format_from_string(parameters["format"].get<std::string>(),YURI_FMT);
	if (format == YURI_FMT_NONE) throw exception::InitializationFailed(
			std::string("Wrong output format ")+parameters["format"].get<std::string>());

	set_output_format(parameters["width"].get<yuri::ssize_t>(),
				parameters["height"].get<yuri::ssize_t>(),
				format);

}

AVScaler::~AVScaler()
{
}

bool AVScaler::set_output_format(int w, int h, long fmt)
{
	yuri::lock_t l(scaler_lock);
	bool changed=false;
	if (w<0 || h<0) {
		scaling_disabled=true;
	} else scaling_disabled=false;
	if (w != w_out) {
		w_out = w;
		changed=true;
	}
	if (h != h_out) {
		h_out = h;
		changed=true;
	}
	if (fmt != format_out) {
		format_out = fmt;
		f_out = av_pixelformat_from_yuri(fmt);
		changed=true;
	}
	if (changed) return do_recheck_conversions();
	return true;
}

bool AVScaler::do_recheck_conversions()
{
	if (!w_in || !w_out || !h_in || !h_out) {// Nothing to do
		scaling=false;
		transforming=false;
		return true;
	}
	if (scaling_disabled) {
		w_out=w_in;
		h_out=h_in;
	}
	if (w_in!=w_out || h_in!=h_out) scaling=true;
	else scaling=false;

	if (format_in != format_out) transforming=true;
	else transforming=false;
	log[log::debug] << "Scaling " << (scaling?"enabled":"disabled") << ", transforming " << (transforming?"enabled":"disabled") << std::endl;
	if (scaling) log[log::debug] << "Scaling " << w_in << "x" << h_in << " => " << w_out << "x" << h_out << std::endl;
	if (transforming) log[log::debug] << "Transforming " << core::BasicPipe::get_format_string(format_in) << " => " << core::BasicPipe::get_format_string(format_out) << std::endl;
	if (f_in == PIX_FMT_NONE) {
		log[log::warning] << "Input format not recognized by libav." << std::endl;
		valid_contexts = false;
		return false;
	}
	if (f_out == PIX_FMT_NONE) {
		log[log::warning] << "Output format not recognized by libav." << std::endl;
		valid_contexts = false;
		return false;
	}
	do_create_contexts();
	return true;
}

void AVScaler::do_create_contexts()
{
	do_delete_contexts();
	log[log::normal] << "[Re]creating contexts" << std::endl;
	if (core::BasicPipe::get_format_group(format_in) != YURI_FMT
			|| core::BasicPipe::get_format_group(format_out) != YURI_FMT) {
		log[log::error] << "Trying to convert unsupported format!" <<std::endl;
		return;
	}
	if (scaling) {
		scale_ctx.reset(sws_getContext(w_in,h_in,f_in, w_out, h_out, f_in, SWS_BICUBIC, 0, 0, 0),av_ctx_deleter);
		frm_inter = allocate_empty_frame(format_in,w_out,h_out);
		pix_inter = convert_to_avpicture(frm_inter);
	}
	if (transforming) {
		transform_ctx.reset(sws_getContext(w_out,h_out,f_in, w_out, h_out, f_out, SWS_BICUBIC, 0, 0, 0),av_ctx_deleter);
		frm_out = allocate_empty_frame(format_out,w_out,h_out);
		pix_out = convert_to_avpicture(frm_out);
	}
	valid_contexts=true;
}

void AVScaler::run()
{
	IOThread::run();
}

core::pBasicFrame AVScaler::scale_frame(const core::pBasicFrame& frame)
{
	yuri::lock_t l(scaler_lock);
	if (!do_prescale_checks()) return core::pBasicFrame();
	if (!do_fetch_frame(frame)) return core::pBasicFrame();
	return do_scale_frame();
}

core::pBasicFrame AVScaler::do_scale_frame()
{
	if (!valid_contexts) return core::pBasicFrame();
	shared_ptr<AVPicture> pic;
	if (synch_frame) {
		pic.reset(new AVPicture);
		for (int i=0;i<4;++i) {
			pic->data[i]=synch_frame->data[i];
			pic->linesize[i]=synch_frame->linesize[i];
		}
	} else {
		pic = convert_to_avpicture(frame);
	}
	if (scaling) {
		sws_scale(scale_ctx.get(),pic->data,pic->linesize,0,h_in,
				pix_inter->data,pix_inter->linesize);
		if (transforming) {
			sws_scale(transform_ctx.get(),pix_inter->data,pix_inter->linesize,0,
					h_out,pix_out->data,pix_out->linesize);
			return do_output_frame(frm_out);
		} else {
			return do_output_frame(frm_inter);
		}
	} else if (transforming) {
		sws_scale(transform_ctx.get(),pic->data,pic->linesize,0,h_out,pix_out->data,pix_out->linesize);
		return do_output_frame(frm_out);
	} //else {
	core::pBasicFrame f = frame;
	frame.reset();
	return f;
//
//		push_raw_video_frame(0,frame);
//		frame.reset();
//	}
//	return;
}


bool AVScaler::do_fetch_frame(const core::pBasicFrame& frame_)
{
//	if (!in[0]) return false;
//	if (in[0]->is_closed()) {
//		close_pipes();
//		return false;
//	}
//	if (in[0]->get_type() != YURI_TYPE_VIDEO) {
//		log[log::debug] << "Connected pipe with type other than video ("
//				<< core::BasicPipe::get_type_string(in[0]->get_type())
//				<< "), that's not gonna work" << std::endl;
//		return false;
//	}
//	if (! (frame= in[0]->pop_frame())) return false;
	frame = frame_;
	if (!frame) return false;
	pts=frame->get_pts();
	duration = frame->get_duration();
	if (do_check_input_frame()) do_recheck_conversions();
	return true;
}

bool AVScaler::do_check_input_frame()
{
	assert(frame);
	bool changed = false;
	if (static_cast<yuri::usize_t>(w_in) != frame->get_width()) {
		w_in = frame->get_width();
		changed=true;
	}
	if (static_cast<yuri::usize_t>(h_in) != frame->get_height()) {
		h_in = frame->get_height();
		changed=true;
	}
	if (format_in != frame->get_format()) {
		format_in = frame->get_format();
		f_in = av_pixelformat_from_yuri(format_in);
		changed=true;
	}
	return changed;
}

void AVScaler::do_delete_contexts()
{
/*	scale_ctx.reset();
	transform_ctx.reset();
	pix_inter.reset();
	frm_inter.reset();
	pix_out.reset();
	frm_out.reset();
*/
}

core::pBasicFrame AVScaler::do_output_frame(core::pBasicFrame frame)
{

	core::pBasicFrame f = frame->get_copy();
	//push_video_frame(0,f,format_out,w_out, h_out, pts, duration, frame->get_dts());
	f->set_parameters(format_out,w_out, h_out);
	f->set_time(pts, duration, frame->get_dts());
	return f;
	//log[log::debug] << "pushed frame " << w_out << "x" << h_out << ", with " <<  f->get_planes_count() << " planes. size: " << f->get_size() << std::endl;
}

bool AVScaler::do_prescale_checks()
{
	if (!out[0]) return false; // We need an output and not full one
	return true;
}

bool AVScaler::synchronous_scale(shared_ptr<AVFrame> fr,int w, int h, PixelFormat fmt, int pts)
{
	yuri::lock_t l(scaler_lock);
	if (w_in!=w || h_in!=h || format_in!=fmt) {
		w_in=w;
		h_in=h;
		format_in=fmt;
		do_recheck_conversions();
	}
	if (!do_prescale_checks()) return false;
	synch_frame=fr;
	this->pts=pts;
	do_scale_frame();
	synch_frame.reset();
	return true;
}
core::pBasicFrame AVScaler::do_simple_single_step(const core::pBasicFrame& frame)
//bool AVScaler::step()
{
	log[log::verbose_debug] << "Step!" << std::endl;
	return scale_frame(frame);
//	return true;
}

void AVScaler::av_ctx_deleter(SwsContext *ctx)
{
	assert(ctx);
	av_free(ctx);
}

std::set<long> AVScaler::get_supported_formats()
{
	std::set<long> fmts;
	fmts.insert(YURI_FMT_RGB);
	fmts.insert(YURI_FMT_RGBA);
	fmts.insert(YURI_FMT_YUV422);
	fmts.insert(YURI_FMT_YUV420_PLANAR);
	fmts.insert(YURI_FMT_YUV422_PLANAR);
	fmts.insert(YURI_FMT_YUV444_PLANAR);
	return fmts;
}
}
}
// End of File
