/*!
 * @file 		TheoraEncoder.cpp
 * @author 		<Your name>
 * @date		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "TheoraEncoder.h"
#include "yuri/core/Module.h"

#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/version.h"


#include <theora/theoradec.h>
namespace yuri {
namespace theora {


IOTHREAD_GENERATOR(TheoraEncoder)

core::Parameters TheoraEncoder::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("TheoraEncoder");
	p["quality"]["Quality of VBR encoder. Set to 0 to lowest quality, 63 for highest."]=6;
	p["bitrate"]["Bitrate for CBR encoding. (probably ignored)"]=16000000;
	p["low_latency"]["Enable low latency mode, this may result in higher bitrates!"]=false;
	p["mux_to_ogg"]["Mux theora packets into ogg bitstream"]=true;
	p["stream_id"]["ID of the stream in OGG bitstream"]=1;
	p["fast_encoding"]["Encoder speed. Set to 0 to max quality, higher values means higher speed (lower CPU load)"]=2;
	return p;

}


TheoraEncoder::TheoraEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("TheoraEncoder")),mux_into_ogg_(true),low_latency_(false),quality_(6),bitrate_(16e6),
stream_id_(1),fast_encoding_(2),frame_duration_(100_ms)
{
	IOTHREAD_INIT(parameters)
	using namespace core::raw_format;
	set_supported_formats({yuv420p, yuv422p, yuv444p});
}

TheoraEncoder::~TheoraEncoder() noexcept
{
}

namespace {
using namespace core::raw_format;
std::map<format_t, th_pixel_fmt> yuri_to_theora_fmt = {
		{yuv422p, TH_PF_422},
		{yuv444p, TH_PF_444},
		{yuv420p, TH_PF_420}
};

bool compare_params(const core::pRawVideoFrame& frame, th_info& info)
{
	auto it = yuri_to_theora_fmt.find(frame->get_format());
	if (it == yuri_to_theora_fmt.end()) return false;
	const auto& res = frame->get_resolution();
	if (info.pic_width != res.width || info.pic_height != res.height) return false;
	if (info.pixel_fmt != it->second) return false;
	return true;
}

bool get_ogg_page(ogg_stream_state& state, ogg_page& page, bool low_latency)
{
	if (!low_latency) {
		return ogg_stream_pageout(&state,&page);
	}
	return ogg_stream_flush(&state,&page);
}

}
void TheoraEncoder::process_packet(ogg_packet& packet)
{
	if (mux_into_ogg_) {
		ogg_stream_packetin(&ogg_state_, &packet);
		ogg_page page;
		while(get_ogg_page(ogg_state_, page, low_latency_)) {
//		while(ogg_stream_flush(&ogg_state_,&page)) {
			auto frame  = core::CompressedVideoFrame::create_empty(core::compressed_frame::ogg, resolution_t{theora_info_.frame_width, theora_info_.frame_height}, page.body_len + page.header_len);
			std::copy(page.header, page.header+page.header_len, frame->data());
			std::copy(page.body, page.body+page.body_len, frame->data()+page.header_len);
			frame->set_duration(frame_duration_); // ????
			push_frame(0,frame);
		}
	} else {
		auto frame  = core::CompressedVideoFrame::create_empty(core::compressed_frame::theora, resolution_t{theora_info_.frame_width, theora_info_.frame_height}, packet.packet, packet.bytes);
		frame->set_duration(frame_duration_); // ??
//		frame->set_timestamp(frame_duration_*packet.packetno)
		push_frame(0,frame);
	}
}


bool TheoraEncoder::init_ctx(const core::pRawVideoFrame& frame)
{
	auto it = yuri_to_theora_fmt.find(frame->get_format());
	if (it == yuri_to_theora_fmt.end()) {
		log[log::error] << "Wrong pixel format!";
		return false;
	}
	const auto& res = frame->get_resolution();

	th_info_init(&theora_info_);
	theora_info_.frame_width = (res.width+15)&~0xF;
	theora_info_.frame_height = (res.height+15)&~0xF;
	theora_info_.pic_width = res.width;
	theora_info_.pic_height = res.height;
	theora_info_.pic_x = 0;
	theora_info_.pic_y = 0;
	theora_info_.pixel_fmt = it->second;
	theora_info_.colorspace = TH_CS_UNSPECIFIED;
	theora_info_.quality=quality_;
	theora_info_.target_bitrate=bitrate_; // 2MB/s

	theora_info_.aspect_numerator = 1;
	theora_info_.aspect_denominator = 1;

	frame_duration_ = frame->get_duration();
	if (frame_duration_ < 1_us) {
		log[log::warning] << "No duration set in incoming frame. Assuming 10fps.";
		frame_duration_ = 100_ms;
	}

	// TODO set fps
	theora_info_.fps_numerator = 1000;
	theora_info_.fps_denominator = frame_duration_.value/(1_ms).value;

	ctx_ = {th_encode_alloc(&theora_info_),[](th_enc_ctx* p){th_encode_free(p);}};
	if (!ctx_) {
		log[log::error] << "Failed to allocate context";
		return false;
	}
	int max_speed = 0;
	th_encode_ctl(ctx_.get(), TH_ENCCTL_GET_SPLEVEL_MAX, &max_speed, sizeof(int));
	fast_encoding_ = std::min(fast_encoding_, max_speed);
	log[log::info] << "Max value for fast encoding is " << max_speed << ", using " << fast_encoding_;
	th_encode_ctl(ctx_.get(), TH_ENCCTL_SET_SPLEVEL, &fast_encoding_, sizeof(int));



	ogg_stream_init(&ogg_state_, stream_id_);

	log[log::info] << "Context allocated";
	std::string v = "Yuri ";
	v+=yuri_version;
	th_comment comment = {nullptr, nullptr, 0, const_cast<char*>(v.c_str())};


	ogg_packet packet;
	while (th_encode_flushheader(ctx_.get(), &comment, &packet)) {
		//log[log::info] << "Got packet with " << packet.bytes << " bytes";
		process_packet(packet);
	}

	return true;
}

namespace {
void set_th_plane(const core::pRawVideoFrame& frame, size_t index, th_img_plane& plane)
{
	//const format_t fmt = frame->get_format();
	const auto res = PLANE_DATA(frame,index).get_resolution();
	using namespace core::raw_format;
	plane.width = res.width;
	plane.height= res.height;
	plane.stride= PLANE_DATA(frame,index).get_line_size();
	plane.data=PLANE_RAW_DATA(frame,index);
}
}

core::pFrame TheoraEncoder::do_special_single_step(const core::pRawVideoFrame& frame)
{
	if (!ctx_ && !init_ctx(frame)) return {};
	if (!compare_params(frame,theora_info_)) return {};

	th_ycbcr_buffer buffer;

	for (int i=0;i<3;++i) {
		set_th_plane(frame,i,buffer[i]);
	}

	int ret;
	if ((ret = th_encode_ycbcr_in(ctx_.get(), buffer))) {
		log[log::warning] << "Failed to encode input data ("<<ret<<")";
	}

	ogg_packet packet;

	while (th_encode_packetout(ctx_.get(), 0, &packet)) {
		process_packet(packet);
	}

	return {};
}
bool TheoraEncoder::set_param(const core::Parameter& param)
{
	if (param.get_name() == "quality") {
		quality_ = param.get<int>();
	} else if (param.get_name() == "bitrate") {
		bitrate_ = param.get<int>();
	} else if (param.get_name() == "low_latency") {
		low_latency_ = param.get<bool>();
	} else if (param.get_name() == "mux_to_ogg") {
		mux_into_ogg_ = param.get<bool>();
	} else if (param.get_name() == "stream_id") {
		stream_id_ = param.get<bool>();
	} else if (param.get_name() == "fast_encoding") {
		fast_encoding_ = param.get<int>();
	}else return base_type::set_param(param);
	return true;
}

} /* namespace theora */
} /* namespace yuri */

