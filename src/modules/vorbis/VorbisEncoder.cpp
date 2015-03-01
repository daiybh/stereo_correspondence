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
	p["bitrate"]["Bitrate of encoded stream in bits (valid for CBR and AVR modes)"]=128000;
	p["quality"]["Quality of encoded VBR stream (valid values -0.1 - 1.0)"]=0.4;
	p["mode"]["Encoding mode (vbr, cbr, abr)"]="cbr";
	p["channels"]["Number of channels to encode. Set to -1 to use number of channels from incoming frames"]=-1;
	return p;
}



namespace {
struct ci_comp {
	bool operator()(const std::string& a, const std::string& b) const {
		return iless(a,b);
	}
};
std::map<std::string, encoding_type_t, ci_comp> encoding_strings = {
		{"cbr", encoding_type_t::cbr},
		{"vbr", encoding_type_t::vbr},
		{"abr", encoding_type_t::abr}
};

}

VorbisEncoder::VorbisEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("vorbis")),initialized_(false),bitrate_(128000),
quality_(0.4),encoding_type_(encoding_type_t::cbr),encoding_channels_(-1)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::raw_audio_format::signed_16bit});
}

VorbisEncoder::~VorbisEncoder() noexcept
{
}

bool VorbisEncoder::initialize(const core::pRawAudioFrame& frame)
{
	if (encoding_channels_ < 0) {
		encoding_channels_ = frame->get_channel_count();
	}

	if (!initialized_) {
		vorbis_info_init(&info_);
		ogg_stream_init(&ogg_state_, /*stream_id_*/10);

		int ret = 0;
		switch (encoding_type_) {
			case encoding_type_t::cbr:
				ret = vorbis_encode_init(&info_, encoding_channels_, frame->get_sampling_frequency(),bitrate_, bitrate_, bitrate_);
				break;
			case encoding_type_t::abr:
				ret = vorbis_encode_init(&info_, encoding_channels_, frame->get_sampling_frequency(),-1, bitrate_, -1);
				break;
			case encoding_type_t::vbr:
				ret = vorbis_encode_init_vbr(&info_, encoding_channels_, frame->get_sampling_frequency(), quality_);
				break;
			default:
				log[log::error] << "Unsupported encoding type";
				return false;
		}
		if (ret) {
			log[log::warning] << "Failed to initialize encoder (" << ret << ")";
			return false;
		}
		ret = vorbis_analysis_init(&state_, &info_);
		if (ret) {
			log[log::warning] << "Failed to initialize analyzer (" << ret << ")";
			return false;
		}
		ret = vorbis_block_init(&state_, &block_);
		if (ret) {
			log[log::warning] << "Failed to prepare block (" << ret << ")";
		}
		vorbis_comment comment;
		vorbis_comment_init(&comment);
		vorbis_comment_add_tag(&comment, "yuri_version", yuri_version);
		vorbis_comment_add_tag(&comment, "encoder", "yuri 2.8");

		ogg_packet packet, packet_comm, packet_code;
		ret = vorbis_analysis_headerout(&state_, &comment, &packet, &packet_comm, &packet_code);
		vorbis_comment_clear(&comment);
		if (ret) {
			log[log::warning] << "Failed to prepare headers (" << ret << ")";
			return false;
		}
		process_packet(packet);
		process_packet(packet_comm);
		process_packet(packet_code);



		initialized_ = true;
		log[log::info] << "Encoder initialized";
	}
	return true;
}

namespace {

template<typename T>
typename std::enable_if<std::numeric_limits<T>::is_signed, float>::type
copy_channel_generic(const void* data, int index)
{
	const T* d = reinterpret_cast<const T*> (data);
	return static_cast<float>(d[index])/std::numeric_limits<T>::max();
}


template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, float>::type
copy_channel_generic(const void* data, int index)
{
	const T* d = reinterpret_cast<const T*> (data);
	return 2.0f*static_cast<float>(d[index])/std::numeric_limits<T>::max()-1.0f;
}


template<format_t fmt>
float copy_channel(const void* data, int index);

template<>
float  copy_channel<core::raw_audio_format::signed_16bit>(const void* data, int index)
{
	return copy_channel_generic<int16_t>(data, index);
}
template<>
float  copy_channel<core::raw_audio_format::signed_32bit>(const void* data, int index)
{
	return copy_channel_generic<int32_t>(data, index);
}
template<>
float  copy_channel<core::raw_audio_format::unsigned_8bit>(const void* data, int index)
{
	return copy_channel_generic<uint8_t>(data, index);
}
template<>
float  copy_channel<core::raw_audio_format::unsigned_16bit>(const void* data, int index)
{
	return copy_channel_generic<uint16_t>(data, index);
}
template<>
float  copy_channel<core::raw_audio_format::unsigned_32bit>(const void* data, int index)
{
	return copy_channel_generic<uint32_t>(data, index);
}
template<>
float  copy_channel<core::raw_audio_format::float_32bit>(const void* data, int index)
{
	const float* d = reinterpret_cast<const float*> (data);
	return d[index];
}


template<format_t fmt>
void copy_channels_process(const core::pRawAudioFrame& frame, float** buffer, int encoding_channels)
{
	const int input_channels = frame->get_channel_count();
	const int copy_channels = std::min(input_channels, encoding_channels);
	const int silence_channels = encoding_channels - copy_channels;
	int index = 0;
	void* data = frame->data();
	for (size_t i =0;i<frame->get_sample_count();++i) {
		for (int ch=0;ch<copy_channels;++ch) {
			buffer[ch][i]=copy_channel<fmt>(data, index+ch);
		}
		index+=input_channels;
	}
	for (size_t i =0;i<frame->get_sample_count();++i) {
			for (int ch=0;ch<silence_channels;++ch) {
				buffer[ch+copy_channels][i]=0.0;
			}
	}
}

void copy_channels_dispatch(const core::pRawAudioFrame& frame, float** buffer, int encoding_channels)
{
	using namespace core::raw_audio_format;
	switch (frame->get_format()) {
		case signed_16bit: copy_channels_process<signed_16bit>(frame, buffer, encoding_channels);
		break;
		case signed_32bit: copy_channels_process<signed_32bit>(frame, buffer, encoding_channels);
		break;
		case unsigned_8bit: copy_channels_process<unsigned_8bit>(frame, buffer, encoding_channels);
		break;
		case unsigned_16bit: copy_channels_process<unsigned_16bit>(frame, buffer, encoding_channels);
		break;
		case unsigned_32bit: copy_channels_process<unsigned_32bit>(frame, buffer, encoding_channels);
		break;
		case float_32bit: copy_channels_process<float_32bit>(frame, buffer, encoding_channels);
		break;
	}
}

}

core::pFrame VorbisEncoder::do_special_single_step(core::pRawAudioFrame frame)
{
	if (!initialize(frame)) return {};

	float** buffer = vorbis_analysis_buffer(&state_, frame->get_sample_count());
//	log[log::info] << "Writing " << frame->get_sample_count() << " samples";

	copy_channels_dispatch(frame, buffer, encoding_channels_);

	vorbis_analysis_wrote(&state_, frame->get_sample_count());

	while (vorbis_analysis_blockout(&state_, &block_) > 0) {
		ogg_packet packet;
		int ret2 = -1;
		if ((ret2 = vorbis_analysis(&block_, nullptr))) {
			log[log::warning] << "Failed to encode packet (" << ret2 << ")";
			continue;
		}
		if ((ret2 = vorbis_bitrate_addblock(&block_))) {
			log[log::warning] << "Failed to abb block to bitrate management  (" << ret2 << ")";
			continue;
		}
		while (vorbis_bitrate_flushpacket(&state_,&packet) > 0) {
			process_packet(packet);
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
	if (param.get_name() == "bitrate") {
		bitrate_ = param.get<int>();
	} else if (param.get_name() == "quality") {
		quality_=param.get<float>();
	} else if (param.get_name() == "mode") {
		auto it = encoding_strings.find(param.get<std::string>());
		if (it == encoding_strings.end()) return false;
		encoding_type_ = it->second;
	} else if (param.get_name() == "channels") {
		encoding_channels_ = param.get<int>();
	} else {
		return base_type::set_param(param);
	}
	return true;
}

} /* namespace vorbis */
} /* namespace yuri */
