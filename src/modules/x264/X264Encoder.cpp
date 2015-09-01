/*!
 * @file 		X264Encoder.cpp
 * @author 		<Your name>
 * @date		28.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "X264Encoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include <cstdarg>
#include <cstdio>
namespace yuri {
namespace x264 {


IOTHREAD_GENERATOR(X264Encoder)

core::Parameters X264Encoder::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("X264Encoder");
	std::string profiles, presets, tunes;
	const char* const * prof = x264_profile_names;
	while (*prof) {
		if (!profiles.empty()) profiles += ", " ;
		profiles += *prof++;
	}
	const char* const * tun = x264_tune_names;
	while (*tun) {
		if (!tunes.empty()) tunes += ", " ;
		tunes += *tun++;
	}
	const char* const * pres = x264_preset_names;
	while (*pres) {
		if (!presets.empty()) presets += ", " ;
		presets += *pres++;
	}

	p["profile"]["X264 profile. Available values: ("+profiles+")"]="baseline";
	p["preset"]["X264 preset. Available values: ("+presets+")"]="ultrafast";
	p["tune"]["X264 tune. Available values: ("+tunes+")"]="zerolatency";
	p["bitrate"]["Average bitrate [KB/s]"]=-1;
	p["max_bitrate"]["Maximum bitrate [KB/s]"]=-1;
	p["cabac"]["Use cabac coder"]=false;
	p["threads"]["Number of threads to use (0 for default)"]=0;
	p["fps"]["Override framerate in incomming frames. (set to 0 to use value specified in frames)"]=fraction_t{25,1};
	return p;
}


namespace {
std::map<format_t, int>supported_formats = {
		{core::raw_format::yuv420p, X264_CSP_I420},
		{core::raw_format::yuv422p, X264_CSP_I422},
		{core::raw_format::yuv444p, X264_CSP_I444}
};

std::map<int, log::debug_flags> x264_log_levels = {
		{X264_LOG_NONE, log::info},
		{X264_LOG_ERROR, log::error},
		{X264_LOG_WARNING, log::warning},
		{X264_LOG_INFO, log::info},
		{X264_LOG_DEBUG, log::debug}
};

void yuri_log(void* priv, int level,  const char *psz, va_list params)
{
	log::Log& log = *reinterpret_cast<log::Log*>(priv);
	char msg[1024];
	int len = std::vsnprintf(msg, 1024, psz, params);
	msg[len]=0;
	// remove trailing newlines, log::Log adds it.
	if (len > 0 && msg[len-1]=='\n') {
		msg[len-1] = 0;
	}
	log::debug_flags l = log::info;
	auto it = x264_log_levels.find(level);
	if (it!=x264_log_levels.end()) l = it->second;
	log[l] << "X264 " << msg;
}

}

X264Encoder::X264Encoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("x264")),encoder_(nullptr),
frame_number_(0),preset_("ultrafast"),tune_("zerolatency"),profile_("baseline"),bitrate_(-1),
max_bitrate_(-1),cabac_(true),threads_(0),
encoded_frames_(0)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(supported_formats);
}

X264Encoder::~X264Encoder() noexcept
{
	if (encoder_) {
		x264_encoder_close(encoder_);
		x264_picture_clean(&picture_in_);
	}
}

core::pFrame X264Encoder::do_special_single_step(core::pRawVideoFrame frame)
{
	auto it = supported_formats.find(frame->get_format());
	if (it == supported_formats.end()) return {};
	const auto res = frame->get_resolution();
	if (!encoder_) {
		x264_param_default_preset(&params_, preset_.c_str(), tune_.c_str());

		params_.i_width = res.width;
		params_.i_height = res.height;
		// TODO: set proper fps
		const auto dur = frame->get_duration();
		fraction_t fps;
		if (fps_.get_value() > 0) {
			fps = fps_;
		} else if (dur > 0.1_us) {
			fps = fraction_t{dur/1_ms, 1000};
		} else {
			fps = fraction_t{25,1};
		}
		log[log::info] << "Using framerate: " << fps;
		params_.i_fps_num = fps.num;
		params_.i_fps_den = fps.denom;
		params_.i_csp = it->second;
		params_.b_repeat_headers = 1;
		params_.pf_log = &yuri_log;
		params_.p_log_private = &log;

		params_.i_threads = threads_;
		params_.b_cabac = cabac_?1:0;
		if (bitrate_ >= 0) {
			params_.rc.i_rc_method = X264_RC_ABR;
			params_.rc.i_bitrate = bitrate_ * 8;

			if (max_bitrate_ >= 0) {
				params_.rc.i_vbv_max_bitrate = max_bitrate_ * 8;
			} else {
				params_.rc.i_vbv_max_bitrate = bitrate_ * 8 * 1.2;
			}
			params_.rc.i_vbv_buffer_size = params_.rc.i_vbv_max_bitrate;
		}

		x264_param_apply_profile(&params_, profile_.c_str());

		x264_picture_alloc(&picture_in_, it->second, res.width, res.height);

		encoder_ = x264_encoder_open(&params_);
	} else {
		if (res.width != static_cast<dimension_t>(params_.i_width) ||
				res.height != static_cast<dimension_t>(params_.i_height)) {
			log[log::warning] << "Frame size changed, ignoring frame";
			return {};
		}

	}
	frame_data_.clear();
	picture_in_.img.i_csp = it->second;
	picture_in_.i_pts = frame_number_++;
	picture_in_.img.i_plane = frame->get_planes_count();
	std::vector<uint8_t*> orig_planes(picture_in_.img.i_plane);
	for (int i = 0;i < picture_in_.img.i_plane; ++i) {
		orig_planes[i]=picture_in_.img.plane[i];
		picture_in_.img.i_stride[i]=PLANE_DATA(frame,i).get_line_size();
		picture_in_.img.plane[i]=PLANE_RAW_DATA(frame,i);
	}
	x264_nal_t* nals;
	int nal_count = 0;
	Timer t0;
	if (x264_encoder_encode(encoder_, &nals, &nal_count, &picture_in_, &picture_out_)) {
		for (int i=0;i<nal_count;++i) {
			process_nal(nals[i]);
		}
	}
	for (int i = 0;i < picture_in_.img.i_plane; ++i) {
		picture_in_.img.plane[i]=orig_planes[i];
	}
	++encoded_frames_;
	encoding_time_+=t0.get_duration();
	if (encoded_frames_ >= 100) {
		log[log::debug] << "Encoding " << encoded_frames_ << " took: " << encoding_time_ << ", that's " << encoding_time_/encoded_frames_ << " per frame";
		encoded_frames_ = 0;
		encoding_time_=0_ms;
	}
	return {core::CompressedVideoFrame::create_empty(core::compressed_frame::h264,
					resolution_t{static_cast<dimension_t>(params_.i_width), static_cast<dimension_t>(params_.i_height)},
					&frame_data_[0], frame_data_.size())};
}


void X264Encoder::process_nal(x264_nal_t& nal)
{
	frame_data_.insert(frame_data_.end(), nal.p_payload, nal.p_payload + nal.i_payload);
}

bool X264Encoder::set_param(const core::Parameter& param)
{
	if(assign_parameters(param)
			(preset_, "preset")
			(tune_, "tune")
			(profile_, "profile")
			(bitrate_, "bitrate")
			(max_bitrate_, "max_bitrate")
			(cabac_, "cabac")
			(threads_, "threads")
			(fps_, "fps"))
		return true;
	return base_type::set_param(param);
}

} /* namespace x264 */
} /* namespace yuri */
