/*
 * RawAVFile.cpp
 *
 *  Created on: Feb 9, 2012
 *      Author: neneko
 */

#include "RawAVFile.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace video {

REGISTER("rawavsource",RawAVFile)
IO_THREAD_GENERATOR(RawAVFile)

core::pParameters RawAVFile::configure()
{
	core::pParameters p = BasicIOThread::configure();
	(*p)["block"]["Threat output pipes as blocking. Specify as max number of frames in output pipe."]=0;
	(*p)["filename"]["File to open"]="";
	(*p)["decode"]["Decode the stream and push out raw video"]=true;
	(*p)["format"]["Format to decode to"]="YUV422";
	(*p)["fps"]["Override framerate. Specify 0 to use original, or negative value to maximal speed."]=0;
	return p;
}

// TODO: number of output streams should be -1 and custom connect_out should be implemented.
RawAVFile::RawAVFile(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		AVCodecBase(_log,parent,"RawAVSource",1,1024),fmtctx(0),block(0),video_stream(0),format_(YURI_FMT_NONE),
		format_out_(YURI_FMT_NONE),decode_(true),fps_(0.0)
{
	IO_THREAD_INIT("rawavsource")
	latency = 10;
	av_register_all();
	if (filename.empty()) throw exception::InitializationFailed("No filename specified!");

	avformat_open_input(&fmtctx, filename.c_str(),0, 0);
	if (!fmtctx) {
		throw exception::InitializationFailed("Failed to allocate Format context");
	}
	if (avformat_find_stream_info(fmtctx,0)<0) {
		log[log::fatal] << "Failed to retrieve stream info!";
		throw exception::InitializationFailed("Failed to retrieve stream info!");
	}
	for (size_t i = 0; i< fmtctx->nb_streams; ++i) {
		if (fmtctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			log[log::debug] << "Found video stream with id " << i << ".";
			if (!video_stream) video_stream=fmtctx->streams[i];
		} else if (fmtctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			log[log::debug] << "Found audio stream with id " << i << ".";
		}
	}
	if (!video_stream) {
		throw exception::InitializationFailed("No video stream in input file!");
	}
	if (!video_stream->codec) {
		throw exception::InitializationFailed("No codec allocated for input file");
	}
	if (!decode_) {
		format_ = yuri_format_from_avcodec(video_stream->codec->codec_id);
		if (format_ == YURI_FMT_NONE) {
			throw exception::InitializationFailed("Unknown format");
		}
		log[log::info] << "Found video stream with format " << core::BasicPipe::get_format_string(format_) <<
		" and resolution " << video_stream->codec->width << "x" << video_stream->codec->height;
	} else {
		video_codec = avcodec_find_decoder(video_stream->codec->codec_id);
		if (!video_codec) {
			throw exception::InitializationFailed("Failed to find decoder");
		}
		if(video_codec->capabilities & CODEC_CAP_TRUNCATED)
			video_stream->codec->flags|=CODEC_FLAG_TRUNCATED;
		if (format_out_ != YURI_FMT_NONE) {
			video_stream->codec->pix_fmt = av_pixelformat_from_yuri(format_out_);
		}

		if (avcodec_open2(video_stream->codec,video_codec,0) < 0) {
			throw exception::InitializationFailed("Failed to open codec!");
		}
		format_ = yuri_format_from_avcodec(video_stream->codec->codec_id);
		format_out_ = yuri_pixelformat_from_av(video_stream->codec->pix_fmt);
		log[log::info] << "Found video stream with format " << core::BasicPipe::get_format_string(format_) <<
			" and resolution " << video_stream->codec->width << "x" << video_stream->codec->height <<
			". Decoding to " << core::BasicPipe::get_format_string(format_out_);
	}


}

RawAVFile::~RawAVFile()
{

}

void RawAVFile::run()
{
	using namespace boost::posix_time;
	IO_THREAD_PRE_RUN
	AVPacket packet;
	av_init_packet(&packet);
	AVFrame *av_frame = avcodec_alloc_frame();
	next_time_ = microsec_clock::local_time();
	time_duration time_delta;
	if (fps_>0) time_delta = microseconds(1e6/fps_);
	else if (fps_ == 0.0) {
		time_delta = microseconds(video_stream->r_frame_rate.den*1e6/video_stream->r_frame_rate.num);
	}
	while (still_running()) {
		if (block) {
			if (out[0] && out[0]->get_size()>=block) {
				sleep(latency);
				continue;
			}
		}
		if (frame) {
			if (fps_>=0) {
				ptime curr_time = microsec_clock::local_time();
				if (curr_time < next_time_) {
					sleep(latency);
					continue;
				}
				next_time_ = next_time_+time_delta;
			}
			push_raw_video_frame(0,frame);
			frame.reset();
			continue;
		}
		if (av_read_frame(fmtctx,&packet)<0) {
			log[log::error] << "Failed to read next packet";
			request_end(YURI_EXIT_FINISHED);
			break;
		}
		if (packet.stream_index != video_stream->index) continue;
		if (!decode_) {
			frame = allocate_frame_from_memory(packet.data, packet.size);
			log[log::debug] << "Pushing packet with size: " << PLANE_SIZE(frame,0);
			size_t dur = packet.duration*video_stream->r_frame_rate.den*1e6/video_stream->r_frame_rate.num;
			if (!dur) dur = video_stream->r_frame_rate.den*1e6/video_stream->r_frame_rate.num;
			size_t pts = 1e3*packet.pts*video_stream->r_frame_rate.den/video_stream->r_frame_rate.num;
			size_t dts = 1e3*packet.dts*video_stream->r_frame_rate.den/video_stream->r_frame_rate.num;
			frame->set_parameters(format_, video_stream->codec->width, video_stream->codec->height);
			frame->set_time(dur, pts, dts);
			log[log::debug] << "Found packet! (pts: " << pts << ", dts: " << dts << ", dur: " << dur;
			log[log::debug] << "num/den:" << video_stream->r_frame_rate.num << "/" << video_stream->r_frame_rate.den;
			log[log::debug] << "orig pts: " << packet.pts << ", dts: " << packet.dts << ", dur: " << packet.duration;
		} else {
			int whole_frame = 0;
			const int height = video_stream->codec->height;
			const int width= video_stream->codec->width;
			int decoded_size = avcodec_decode_video2(video_stream->codec,av_frame, &whole_frame,&packet);
			if (decoded_size < 0) {
				log[log::warning] << "Failed to decode frame";
				continue;
			}
			if (decoded_size != packet.size) {
				log[log::warning] << "Used only " << decoded_size << " bytes out of " << packet.size;
			}
			//assert(format_out_);
			format_t fmt = yuri_pixelformat_from_av(static_cast<PixelFormat>(av_frame->format));
			if (format_out_ != fmt) {
				log[log::warning] << "Unexpected frame format! Expected '" << core::BasicPipe::get_format_string(format_out_)
				<< "', but got '" << core::BasicPipe::get_format_string(fmt) << "'";
				format_out_ = fmt;
			}
			if (format_out_ == YURI_FMT_NONE) continue;
			frame = allocate_empty_frame(format_out_,width, height, true);
			FormatInfo_t fi = core::BasicPipe::get_format_info(format_out_);
			for (int i=0;i<4;++i) {
				if ((av_frame->linesize[i] == 0) || (!av_frame->data[i])) break;
				if (i >= frame->get_planes_count()) {
					log[log::warning] << "BUG? Inconsistent number of planes";
					break;
				}
				size_t line_size = width/fi->plane_x_subs[i];
				size_t lines = height/fi->plane_y_subs[i];
				assert(line_size <= static_cast<yuri::size_t>(av_frame->linesize[i]));
				//assert(av_frame->linesize[i]*height <= PLANE_SIZE(frame,i));
				for (int l=0;l<lines;++l) {
					std::copy(av_frame->data[i]+l*av_frame->linesize[i],
								av_frame->data[i]+l*av_frame->linesize[i]+line_size,
								PLANE_RAW_DATA(frame,i)+l*line_size);
				}
			}
			//push_raw_video_frame(0,frame);
		}
	}
	av_free(av_frame);
	av_free_packet(&packet);
	IO_THREAD_POST_RUN
}

bool RawAVFile::set_param(const core::Parameter &parameter)
{
	if (parameter.name == "block") {
		block=parameter.get<yuri::size_t>();
	} else if (parameter.name == "filename") {
		filename=parameter.get<std::string>();
	} else if (parameter.name == "decode") {
		decode_=parameter.get<bool>();
	} else if (parameter.name == "format") {
		format_out_=core::BasicPipe::get_format_from_string(parameter.get<std::string>());
	} else if (parameter.name == "fps") {
		fps_ = parameter.get<double>();
	} else {
		return AVCodecBase::set_param(parameter);
	}
	return true;

}

} /* namespace video */
} /* namespace yuri */
