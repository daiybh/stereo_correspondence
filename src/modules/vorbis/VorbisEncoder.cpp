/*!
 * @file 		VorbisEncoder.cpp
 * @author 		<Your name>
 * @date		25.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "VorbisEncoder.h"
#include "yuri/core/Module.h"
#include "yuri/version.h"

#include "yuri/core/frame/raw_audio_frame_types.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_types.h"

namespace yuri {
namespace vorbis {


IOTHREAD_GENERATOR(VorbisEncoder)


core::Parameters VorbisEncoder::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("VorbisEncoder");
	return p;
}


VorbisEncoder::VorbisEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("vorbis")),initialized_(false)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::raw_audio_format::signed_16bit});
}

VorbisEncoder::~VorbisEncoder() noexcept
{
}

core::pFrame VorbisEncoder::do_special_single_step(const core::pRawAudioFrame& frame)
{
	if (frame->get_channel_count() != 2) {
		log[log::warning] << "Only stereo samples supported atm";
		return {};
	}
	if (!initialized_) {
		vorbis_info_init(&info_);
		ogg_stream_init(&ogg_state_, /*stream_id_*/10);

		int ret = vorbis_encode_init(&info_, frame->get_channel_count(), frame->get_sampling_frequency(),128000,128000,128000);
//		int ret = vorbis_encode_init(&info_,frame->get_channel_count(), frame->get_sampling_frequency(),-1,128000,-1);
//		int ret = vorbis_encode_init_vbr(&info_,frame->get_channel_count(), frame->get_sampling_frequency(),.4);
		if (ret) {
			log[log::warning] << "Failed to initialize encoder (" << ret << ")";
			return {};
		}
		ret = vorbis_analysis_init(&state_, &info_);
		if (ret) {
			log[log::warning] << "Failed to initialize analyzer (" << ret << ")";
			return {};
		}
		ret = vorbis_block_init(&state_, &block_);
		if (ret) {
			log[log::warning] << "Failed to prepare block (" << ret << ")";
		}
		vorbis_comment comment;
		vorbis_comment_init(&comment);
		vorbis_comment_add_tag(&comment, "yuri_version", yuri_version);

		ogg_packet packet, packet_comm, packet_code;
		ret = vorbis_analysis_headerout(&state_, &comment, &packet, &packet_comm, &packet_code);
		vorbis_comment_clear(&comment);
		if (ret) {
			log[log::warning] << "Failed to prepare headers (" << ret << ")";
			return {};
		}
		process_packet(packet);
		process_packet(packet_comm);
		process_packet(packet_code);



		initialized_ = true;
		log[log::info] << "Encoder initialized";
	}

	float** buffer = vorbis_analysis_buffer(&state_, frame->get_sample_count());
	int16_t* data = reinterpret_cast<int16_t*>(frame->data());
	//std::transform(data, data+frame->get_sample_count()*frame->get_channel_count(), *buffer, [](const int16_t& v){return static_cast<float>(v)/32768;});
	for (size_t i =0;i<frame->get_sample_count();++i) {
		buffer[0][i]=static_cast<float>(data[i*2])/32768.0;
		buffer[1][i]=static_cast<float>(data[i*2+1])/32768.0;
	}
	vorbis_analysis_wrote(&state_, frame->get_sample_count());

	int ret = 0;
	while (vorbis_analysis_blockout(&state_, &block_) > 0) {
		ogg_packet packet;
//		if ((ret = vorbis_analysis(&block_, &packet))) {
//			log[log::warning] << "Failed to encode packet (" << ret << ")";
//		} else {
//			process_packet(packet);
//		}
		if ((ret = vorbis_analysis(&block_, nullptr))) {
			log[log::warning] << "Failed to encode packet (" << ret << ")";
			continue;
		}
		if ((ret = vorbis_bitrate_addblock(&block_))) {
			log[log::warning] << "Failed to abb block to bitrate management  (" << ret << ")";
			continue;
		}
		while ((ret = vorbis_bitrate_flushpacket(&state_,&packet)) >= 0) {
			process_packet(packet);
			if (ret == 0) break;
		}

	}
	return {};
}
namespace {
bool get_ogg_page(ogg_stream_state& state, ogg_page& page, bool low_latency)
{
	if (!low_latency) {
		return ogg_stream_pageout(&state,&page);
	}
	return ogg_stream_flush(&state,&page);
}
}

bool VorbisEncoder::process_packet(ogg_packet& packet)
{
	ogg_stream_packetin(&ogg_state_, &packet);
	ogg_page page;
	while(get_ogg_page(ogg_state_, page, /*low_latency_*/ false)) {
		uvector<uint8_t> data (page.body_len + page.header_len);
		std::copy(page.header, page.header+page.header_len, data.data());
		std::copy(page.body, page.body+page.body_len, data.data()+page.header_len);
// !!!!!!!!!!!! TODO This obviously should not produce compressed video....
		auto frame  = core::CompressedVideoFrame::create_empty(core::compressed_frame::ogg, resolution_t{0,0}, data.data(), data.size());
//					frame->set_duration(frame_duration_); // ????
		push_frame(0,frame);
	}
	return true;
}

bool VorbisEncoder::set_param(const core::Parameter& param)
{
	return base_type::set_param(param);
}

} /* namespace vorbis */
} /* namespace yuri */
