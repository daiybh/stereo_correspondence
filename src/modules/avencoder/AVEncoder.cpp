#include "AVEncoder.h"

namespace yuri
{

namespace video
{



REGISTER("avencoder",AVEncoder)

IO_THREAD_GENERATOR(AVEncoder)

shared_ptr<Parameters> AVEncoder::configure()
{
	shared_ptr<Parameters> p(new Parameters());
	(*p)["codec"]="";
	(*p)["width"]=640;
	(*p)["height"]=480;
	(*p)["bps"]=2*1048576;
	(*p)["fps"]=25;
	(*p)["buffer_size"]=10485760;

	p->set_max_pipes(1,1);
	yuri::format_t fmt1, fmt2;
	BOOST_FOREACH(fmt1,get_supported_input_formats()) {
		p->add_input_format(fmt1);
		BOOST_FOREACH(fmt2,get_supported_output_formats()) {
			p->add_converter(fmt1,fmt2,0,false);
		}
	}
	BOOST_FOREACH(fmt2,get_supported_output_formats()) {
		p->add_output_format(fmt2);
	}
	return p;
}

AVEncoder::AVEncoder(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		AVCodecBase(_log,parent,"Encoder"),buffer_size(1048576)
{
	IO_THREAD_INIT("AVEncoder")
	if (codec_id == CODEC_ID_NONE) throw InitializationFailed(string("Unknown/unsupported codec"));
	buffer.reset(new uint8_t[buffer_size]);
	if (!init_encoder())
		throw InitializationFailed(string("Failed to init encoder"));
}

AVEncoder::~AVEncoder()
{
}

bool AVEncoder::init_encoder()
{
	//this->codec_id=codec_id;
	//log[debug]<<"Looking for encoder" << std::endl;
	if (!find_encoder()) return false;
	//this->width = width;
	//this->height = height;
	//log[debug]<<"Encoder found" << std::endl;
	if (c) log[debug]<<"Codec found" << std::endl;
	if (cc) log[debug]<<"We also have codec context" << std::endl;
	//this->cc->time_base= (AVRational){1,25};
	log[info] << "Selected encoder " << c->long_name << endl;
	const PixelFormat *p = c->pix_fmts;
	supported_formats_for_current_codec.clear();
	yuri::format_t fmt;
	while (*p!=PIX_FMT_NONE) {
		fmt = yuri_pixelformat_from_av(*p);
		log[info] << "Codec supports format " << avcodec_get_pix_fmt_name(*p) <<
				(fmt==YURI_FMT_NONE?" [Not supported in libyuri]":" [OK]") << endl;
		if (fmt!=YURI_FMT_NONE) {
			log[info] << "\tMaps to libyuri format " << BasicPipe::get_format_string(fmt)<<endl;;
			supported_formats_for_current_codec.insert(fmt);
		}
		*p++;
	}
	const AVProfile *prof = c->profiles;
	if (!prof) {
		log[info] << "Selected codec does not support profiles" << endl;
	} else {
		while (prof->profile != FF_PROFILE_UNKNOWN) {
			log[info] << "Supported profile " << prof->profile << ": " << prof->name << endl;
			prof++;
		}
	}
	if (!init_codec(AVMEDIA_TYPE_VIDEO, width, height, bps, fps, 1)) return false;
	time_step=(float)cc->time_base.num/(float)cc->time_base.den;
	return true;
}

void AVEncoder::run()
{
	/*while(still_running()) {

		usleep(1000);
	}*/
	BasicIOThread::run();
	//close_pipes();
}

void AVEncoder::encode_frame()
{
	if (!c || !cc || !in[0].get() || !out[0].get()) return;
	shared_ptr<BasicFrame> f = in[0]->pop_frame();
	if (!f) return;
	assert(buffer.get());
	if (!supported_formats_for_current_codec.count(f->get_format())) {
		log[warning] << "Frame format " << BasicPipe::get_format_string(f->get_format()) << " not supported in actual codec" << endl;
		return;
	}
	if (current_format != f->get_format()) {
		log[warning] << "Input format changed, trying to reinitialize codec for this format" << endl;
		PixelFormat newf = av_pixelformat_from_yuri(f->get_format());
		// VERY ugly hack
		if (codec_id == CODEC_ID_MJPEG /*|| codec_id==CODEC_ID_H264*/) {
			if (newf == PIX_FMT_YUV422P) newf = PIX_FMT_YUVJ422P;
			else if (newf == PIX_FMT_YUV420P) newf = PIX_FMT_YUVJ420P;
		}
		cc->pix_fmt= newf;
		if (!init_codec(AVMEDIA_TYPE_VIDEO, width, height, bps, fps, 1)) return;
	}
	frame = convert_to_avframe(f);
	frame->pts = f->get_pts() * cc->time_base.den / cc->time_base.num;
//	log[debug] << "Prepared frame with linesizes " << frame->linesize[0] << " and " << frame->linesize[1] << endl;

	int used=avcodec_encode_video(cc,buffer.get(),buffer_size,frame.get());
	if (used < 0 || (yuri::size_t)used>buffer_size) {
		log[warning] << "Encoding probably failed, used " << used << "Bytes" << std::endl;
		return;
	}
	yuri::size_t pts, dts, duration;

	pts = cc->coded_frame->display_picture_number * cc->time_base.num * 1e6 / cc->time_base.den;
	dts = cc->coded_frame->coded_picture_number * cc->time_base.num * 1e6 / cc->time_base.den;
	duration = cc->time_base.num * 1e6 / cc->time_base.den;
	log[verbose_debug] << "Encoded frame with pts: " << pts << ", dpn: " <<
				cc->coded_frame->display_picture_number << ", cpn: " <<
				cc->coded_frame->coded_picture_number << "with ratio: " <<
				cc->time_base.num << "/" << cc->time_base.den << endl;
	shared_ptr<BasicFrame> out_frame = allocate_frame_from_memory(buffer.get(),used);
	push_video_frame(0,out_frame,yuri_format_from_avcodec(this->codec_id),width,height,pts,duration,dts);
}

bool AVEncoder::step()
{
	encode_frame();
	return true;
}

set<yuri::format_t> AVEncoder::get_supported_input_formats()
{
	set<yuri::format_t> fmts;
	fmts.insert(YURI_FMT_RGB);
	fmts.insert(YURI_FMT_RGBA);
	fmts.insert(YURI_FMT_YUV422);
	fmts.insert(YURI_FMT_YUV420_PLANAR);
	fmts.insert(YURI_FMT_YUV422_PLANAR);
	fmts.insert(YURI_FMT_YUV444_PLANAR);
	return fmts;
}
set<yuri::format_t> AVEncoder::get_supported_output_formats()
{
	set<yuri::format_t> fmts;
	fmts.insert(YURI_VIDEO_MPEG2);
	fmts.insert(YURI_VIDEO_MPEG1);
	fmts.insert(YURI_VIDEO_MJPEG);
	fmts.insert(YURI_VIDEO_DV);
	fmts.insert(YURI_VIDEO_HUFFYUV);
	fmts.insert(YURI_VIDEO_H264);
	return fmts;

}

bool AVEncoder::set_param(Parameter &param)
{
	if (param.name == "buffer_size") {
		buffer_size = param.get<yuri::size_t>();
	} else if (param.name == "width") {
		width = param.get<yuri::size_t>();
	} else if (param.name == "height") {
		height = param.get<yuri::size_t>();
	} else if (param.name == "bps") {
		bps = param.get<yuri::size_t>();
	} else if (param.name == "fps") {
		fps = param.get<yuri::size_t>();
	} else if (param.name == "codec") {
		codec_id = get_codec_from_string(param.get<string>());
	} else

		return BasicIOThread::set_param(param);
	return true;

}
}

}
