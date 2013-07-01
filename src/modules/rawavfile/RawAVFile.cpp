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
	(*p)["max_video"]["Maximal number of video streams to process"]=1;
	(*p)["max_audio"]["Maximal number of audio streams to process"]=1;
	return p;
}

// TODO: number of output streams should be -1 and custom connect_out should be implemented.
RawAVFile::RawAVFile(log::Log &_log, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		AVCodecBase(_log,parent,"RawAVSource",1,1024),fmtctx(0),block(0),video_format_out_(YURI_FMT_NONE),
		decode_(true),fps_(0.0),max_video_streams_(1),max_audio_streams_(1)
{
	IO_THREAD_INIT("rawavsource")
	latency = 10;
	resize(0,max_video_streams_+max_audio_streams_);
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
			//if (!video_stream) video_stream=fmtctx->streams[i];
			if (video_streams_.size() < max_video_streams_ && fmtctx->streams[i]->codec) {
				video_streams_.push_back(fmtctx->streams[i]);
			}
		} else if (fmtctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			log[log::debug] << "Found audio stream with id " << i << ".";
			if (audio_streams_.size() < max_audio_streams_ && fmtctx->streams[i]->codec) {
				audio_streams_.push_back(fmtctx->streams[i]);
			}
		}
	}
	if (video_streams_.empty() && audio_streams_.empty()) {
		throw exception::InitializationFailed("No stream in input file!");
	}
//	if (!video_stream->codec) {
//		throw exception::InitializationFailed("No codec allocated for input file");
//	}
	video_formats_out_.resize(video_streams_.size(),video_format_out_);
	video_codecs_.resize(video_streams_.size(),0);
	video_formats_.resize(video_streams_.size(),YURI_FMT_NONE);
	frames_.resize(video_streams_.size());
	if (!decode_) {
		for (size_t i=0;i<video_streams_.size();++i) {
			video_formats_[i] = yuri_format_from_avcodec(video_streams_[i]->codec->codec_id);
			if (video_formats_[i] == YURI_FMT_NONE) {
				throw exception::InitializationFailed("Unknown format");
			}
			log[log::info] << "Found video stream with format " << core::BasicPipe::get_format_string(video_formats_[i]) <<
			" and resolution " << video_streams_[i]->codec->width << "x" << video_streams_[i]->codec->height;
		}
	} else {
		for (size_t i=0;i<video_streams_.size();++i) {
			video_codecs_[i] = avcodec_find_decoder(video_streams_[i]->codec->codec_id);
			if (!video_codecs_[i]) {
				throw exception::InitializationFailed("Failed to find decoder");
			}
			if(video_codecs_[i]->capabilities & CODEC_CAP_TRUNCATED)
				video_streams_[i]->codec->flags|=CODEC_FLAG_TRUNCATED;
			if (video_formats_out_[i] != YURI_FMT_NONE) {
				video_streams_[i]->codec->pix_fmt = av_pixelformat_from_yuri(video_formats_out_[i]);
			}

			if (avcodec_open2(video_streams_[i]->codec,video_codecs_[i],0) < 0) {
				throw exception::InitializationFailed("Failed to open codec!");
			}
			video_formats_[i] = yuri_format_from_avcodec(video_streams_[i]->codec->codec_id);
			video_formats_out_[i] = yuri_pixelformat_from_av(video_streams_[i]->codec->pix_fmt);
			log[log::info] << "Found video stream with format " << core::BasicPipe::get_format_string(video_formats_[i]) <<
				" and resolution " << video_streams_[i]->codec->width << "x" << video_streams_[i]->codec->height <<
				". Decoding to " << core::BasicPipe::get_format_string(video_formats_out_[i]);
		}
	}


}

RawAVFile::~RawAVFile()
{

}

void RawAVFile::run()
{
//	using namespace boost::posix_time;
	IO_THREAD_PRE_RUN
	AVPacket packet;
	av_init_packet(&packet);
	AVFrame *av_frame = avcodec_alloc_frame();
	next_times_.resize(video_streams_.size(),std::chrono::steady_clock::now());
	std::vector<time_duration> time_deltas(video_streams_.size());
	for (size_t i=0;i<video_streams_.size();++i) {
		if (fps_>0) time_deltas[i] = nanoseconds(static_cast<nanoseconds::rep>(1e9/fps_));
		else if (fps_ == 0.0) {
			time_deltas[i] = nanoseconds(static_cast<nanoseconds::rep>(video_streams_[i]->r_frame_rate.den*1e9/video_streams_[i]->r_frame_rate.num));
		}
	}
	while (still_running()) {
		if (block) {
			bool block_all = true;
			for (size_t i=0;i<video_streams_.size();++i) {
				if (out[i] && out[i]->get_size()<block) {
					block_all = false;
					break;
				}
			}
			if (block_all) {
				sleep(latency);
				continue;
			}
		}
		bool ready = false;
		for (size_t i=0;i<video_streams_.size();++i) {
			if (frames_[i]) {
				if (fps_>=0) {
					time_value curr_time = std::chrono::steady_clock::now();
					if (curr_time < next_times_[i]) {
						continue;
					} else {
						next_times_[i] = next_times_[i]+time_deltas[i];
						ready = true;
					}
				}
				push_raw_video_frame(i,frames_[i]);
				frames_[i].reset();
				ready=true;
//				continue;
			} else {
				ready = true;
			}
		}
		if (!ready) {
			sleep(latency);
			continue;
		}
		if (av_read_frame(fmtctx,&packet)<0) {
			log[log::error] << "Failed to read next packet";
			request_end(YURI_EXIT_FINISHED);
			break;
		}
		size_t idx=max_video_streams_+1;
		for (size_t i=0;i<video_streams_.size();++i) {
			if (packet.stream_index == video_streams_[i]->index) {
				idx = i;
				break;
			}
		}
		if (idx>max_video_streams_) continue;
		if (!decode_) {
			frames_[idx] = allocate_frame_from_memory(packet.data, packet.size);
			log[log::debug] << "Pushing packet with size: " << PLANE_SIZE(frames_[idx],0);
			size_t dur = packet.duration*video_streams_[idx]->r_frame_rate.den*1e6/video_streams_[idx]->r_frame_rate.num;
			if (!dur) dur = video_streams_[idx]->r_frame_rate.den*1e6/video_streams_[idx]->r_frame_rate.num;
			size_t pts = 1e3*packet.pts*video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;
			size_t dts = 1e3*packet.dts*video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;
			frames_[idx]->set_parameters(video_formats_[idx], video_streams_[idx]->codec->width, video_streams_[idx]->codec->height);
			frames_[idx]->set_time(dur, pts, dts);
			log[log::debug] << "Found packet! (pts: " << pts << ", dts: " << dts << ", dur: " << dur;
			log[log::debug] << "num/den:" << video_streams_[idx]->r_frame_rate.num << "/" << video_streams_[idx]->r_frame_rate.den;
			log[log::debug] << "orig pts: " << packet.pts << ", dts: " << packet.dts << ", dur: " << packet.duration;
		} else {
			int whole_frame = 0;
			const int height = video_streams_[idx]->codec->height;
			const int width= video_streams_[idx]->codec->width;
			int decoded_size = avcodec_decode_video2(video_streams_[idx]->codec,av_frame, &whole_frame,&packet);
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
			frames_[idx] = allocate_empty_frame(format_out_,width, height, true);
			FormatInfo_t fi = core::BasicPipe::get_format_info(format_out_);
			for (size_t i=0;i<4;++i) {
				if ((av_frame->linesize[i] == 0) || (!av_frame->data[i])) break;
				if (i >= frames_[idx]->get_planes_count()) {
					log[log::warning] << "BUG? Inconsistent number of planes";
					break;
				}
				size_t line_size = width/fi->plane_x_subs[i];
				size_t lines = height/fi->plane_y_subs[i];
				assert(line_size <= static_cast<yuri::size_t>(av_frame->linesize[i]));
				//assert(av_frame->linesize[i]*height <= PLANE_SIZE(frame,i));
				for (size_t l=0;l<lines;++l) {
					std::copy(av_frame->data[i]+l*av_frame->linesize[i],
								av_frame->data[i]+l*av_frame->linesize[i]+line_size,
								PLANE_RAW_DATA(frames_[idx],i)+l*line_size);
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
		video_format_out_=core::BasicPipe::get_format_from_string(parameter.get<std::string>());
	} else if (parameter.name == "fps") {
		fps_ = parameter.get<double>();
	} else if (parameter.name == "max_video") {
		max_video_streams_ = parameter.get<size_t>();
	} else if (parameter.name == "max_audio") {
		max_audio_streams_ = parameter.get<double>();
	} else {
		return AVCodecBase::set_param(parameter);
	}
	return true;

}

} /* namespace video */
} /* namespace yuri */
