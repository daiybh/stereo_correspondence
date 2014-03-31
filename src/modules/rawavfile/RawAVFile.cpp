/*
 * RawAVFile.cpp
 *
 *  Created on: Feb 9, 2012
 *      Author: neneko
 */

#include "RawAVFile.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include <cassert>
namespace yuri {
namespace rawavfile {

IOTHREAD_GENERATOR(RawAVFile)

MODULE_REGISTRATION_BEGIN("rawavfile")
	REGISTER_IOTHREAD("rawavsource",RawAVFile)
MODULE_REGISTRATION_END()


namespace {
	const std::string unknown_format = "Unknown";
	const std::string& get_format_name_no_throw(format_t fmt) {
		try {
			return core::raw_format::get_format_name(fmt);
		}
		catch(std::exception&){}
		try {
			return core::compressed_frame::get_format_name(fmt);
		}
		catch(std::exception&){}
		return unknown_format;
	}
}

core::Parameters RawAVFile::configure()
{
	core::Parameters p = IOThread::configure();
	p["block"]["Threat output pipes as blocking. Specify as max number of frames in output pipe."]=0;
	p["filename"]["File to open"]="";
	p["decode"]["Decode the stream and push out raw video"]=true;
	p["format"]["Format to decode to"]="YUV422";
	p["fps"]["Override framerate. Specify 0 to use original, or negative value to maximal speed."]=0;
	p["max_video"]["Maximal number of video streams to process"]=1;
	p["max_audio"]["Maximal number of audio streams to process"]=1;
	return p;
}

// TODO: number of output streams should be -1 and custom connect_out should be implemented.
RawAVFile::RawAVFile(const log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters)
	:IOThread(_log,parent,0,1024,"RawAVSource"),
	fmtctx(nullptr),block(0),video_format_out_(0),
	decode_(true),fps_(0.0),max_video_streams_(1),max_audio_streams_(1)
{
	IOTHREAD_INIT(parameters)
	set_latency (10_us);
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
	video_formats_.resize(video_streams_.size(),0);
	frames_.resize(video_streams_.size());
	if (!decode_) {
		for (size_t i=0;i<video_streams_.size();++i) {
			video_formats_[i] = libav::yuri_format_from_avcodec(video_streams_[i]->codec->codec_id);
			if (video_formats_[i] == 0) {
				throw exception::InitializationFailed("Unknown format");
			}
			log[log::info] << "Found video stream with format " << get_format_name_no_throw(video_formats_[i]) <<
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
			if (video_formats_out_[i] != 0) {
				video_streams_[i]->codec->pix_fmt = libav::avpixelformat_from_yuri(video_formats_out_[i]);
			}

			if (avcodec_open2(video_streams_[i]->codec,video_codecs_[i],0) < 0) {
				throw exception::InitializationFailed("Failed to open codec!");
			}
			video_formats_[i] = libav::yuri_format_from_avcodec(video_streams_[i]->codec->codec_id);
			video_formats_out_[i] = libav::yuri_pixelformat_from_av(video_streams_[i]->codec->pix_fmt);
			log[log::info] << "Found video stream with format " << get_format_name_no_throw(video_formats_[i]) <<
				" and resolution " << video_streams_[i]->codec->width << "x" << video_streams_[i]->codec->height <<
				". Decoding to " << get_format_name_no_throw(video_formats_out_[i]);
		}
	}


}

RawAVFile::~RawAVFile() noexcept
{

}

void RawAVFile::run()
{
//	using namespace boost::posix_time;
//	IO_THREAD_PRE_RUN
	AVPacket packet;
	av_init_packet(&packet);
	bool keep_packet = false;
	AVFrame *av_frame = avcodec_alloc_frame();
	next_times_.resize(video_streams_.size(),timestamp_t{});
	std::vector<duration_t> time_deltas(video_streams_.size());

	for (size_t i=0;i<video_streams_.size();++i) {
		if (fps_>0) time_deltas[i] = 1_s/fps_;
		else if (fps_ == 0.0) {
			time_deltas[i] = 1_s*video_streams_[i]->r_frame_rate.den/video_streams_[i]->r_frame_rate.num;
			log[log::info] << "Delta " << i << " " << time_deltas[i];
		}
	}


	while (still_running()) {
		// TODO reimplement blocking!!
		/*if (block) {
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
		}*/
		bool ready = false;
		for (size_t i=0;i<video_streams_.size();++i) {
			if (frames_[i]) {
				if (fps_>=0) {
					timestamp_t curr_time;//= std::chrono::steady_clock::now();
					if (curr_time < next_times_[i]) {
//						log[log::info] << "Not yet... " << i << ", remaining " <<  (next_times_[i] - curr_time);
						continue;
					} else {
						next_times_[i] = next_times_[i]+time_deltas[i];
						//ready = true;
					}
				}
//				log[log::info] << "Pushing... " << i;
				push_frame(i,frames_[i]);
				frames_[i].reset();
				ready=true;
//				continue;
			} else {
				ready = true;
			}
		}
		if (!ready) {
			sleep(get_latency());
			continue;
		}
		if (!keep_packet && av_read_frame(fmtctx,&packet)<0) {
			log[log::error] << "Failed to read next packet";
			request_end(core::yuri_exit_finished);
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
			core::pCompressedVideoFrame f = core::CompressedVideoFrame::create_empty(video_formats_[idx],
					resolution_t{static_cast<dimension_t>(video_streams_[idx]->codec->width), static_cast<dimension_t>(video_streams_[idx]->codec->height)},
					packet.data,
					packet.size	);
//			frames_[idx] = allocate_frame_from_memory(packet.data, packet.size);
			frames_[idx] = f;
			log[log::debug] << "Pushing packet with size: " << f->size();
			duration_t dur = 1_s * packet.duration*video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;
			if (!dur.value) dur = 1_s * video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;

//			size_t pts = 1e3*packet.pts*video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;
//			size_t dts = 1e3*packet.dts*video_streams_[idx]->r_frame_rate.den/video_streams_[idx]->r_frame_rate.num;


//			frames_[idx]->set_parameters(video_formats_[idx], video_streams_[idx]->codec->width, video_streams_[idx]->codec->height);
//			frames_[idx]->set_time(dur, pts, dts);
			f->set_duration(dur);


			log[log::debug] << "Found packet!"/* (pts: " << pts << ", dts: " << dts <<*/ ", dur: " << dur;
			log[log::debug] << "num/den:" << video_streams_[idx]->r_frame_rate.num << "/" << video_streams_[idx]->r_frame_rate.den;
			log[log::debug] << "orig pts: " << packet.pts << ", dts: " << packet.dts << ", dur: " << packet.duration;
		} else {
			int whole_frame = 0;

			keep_packet = false;
			int decoded_size = avcodec_decode_video2(video_streams_[idx]->codec,av_frame, &whole_frame,&packet);
			if (!whole_frame) {
//				log[log::warning] << "No frame this time...";
				continue;
			}
			if (decoded_size < 0) {
				log[log::warning] << "Failed to decode frame";
				continue;
			}

			if (decoded_size != packet.size) {
				keep_packet = true;
				log[log::debug] << "Used only " << decoded_size << " bytes out of " << packet.size;
			}

			auto f = libav::yuri_frame_from_av(*av_frame);
			if (!f) continue;
			if (format_out_ != f->get_format()) {
				log[log::warning] << "Unexpected frame format! Expected '" << get_format_name_no_throw(format_out_)
				<< "', but got '" << get_format_name_no_throw(f->get_format()) << "'";
				format_out_ = f->get_format();
			}

			frames_[idx] = f;
		}
	}
	av_free(av_frame);
	av_free_packet(&packet);
//	IO_THREAD_POST_RUN
}

bool RawAVFile::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name() == "block") {
		block=parameter.get<yuri::size_t>();
	} else if (parameter.get_name() == "filename") {
		filename=parameter.get<std::string>();
	} else if (parameter.get_name() == "decode") {
		decode_=parameter.get<bool>();
	} else if (parameter.get_name() == "format") {
		video_format_out_=core::raw_format::parse_format(parameter.get<std::string>());
	} else if (parameter.get_name() == "fps") {
		fps_ = parameter.get<double>();
	} else if (parameter.get_name() == "max_video") {
		max_video_streams_ = parameter.get<size_t>();
	} else if (parameter.get_name() == "max_audio") {
		max_audio_streams_ = parameter.get<double>();
	} else {
		return IOThread::set_param(parameter);
	}
	return true;

}

} /* namespace video */
} /* namespace yuri */

