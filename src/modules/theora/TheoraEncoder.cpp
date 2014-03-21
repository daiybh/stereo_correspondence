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
namespace yuri {
namespace theora {


IOTHREAD_GENERATOR(TheoraEncoder)

core::Parameters TheoraEncoder::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("TheoraEncoder");
	return p;

}


TheoraEncoder::TheoraEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("TheoraEncoder"))
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


}
void TheoraEncoder::process_packet(ogg_packet& packet)
{
	ogg_stream_packetin(&ogg_state_, &packet);
	ogg_page page;
	while(ogg_stream_pageout(&ogg_state_,&page)) {
		auto frame  = core::CompressedVideoFrame::create_empty(core::compressed_frame::ogg, resolution_t{theora_info_.frame_width, theora_info_.frame_height}, page.body_len + page.header_len);
		std::copy(page.header, page.header+page.header_len, frame->data());
		std::copy(page.body, page.body+page.body_len, frame->data()+page.header_len);
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

	theora_info_.frame_width = (res.width+15)&~0xF;
	theora_info_.frame_height = (res.height+15)&~0xF;
	theora_info_.pic_width = res.width;
	theora_info_.pic_height = res.height;
	theora_info_.pic_x = 0;
	theora_info_.pic_y = 0;
	theora_info_.pixel_fmt = it->second;
	theora_info_.colorspace = TH_CS_UNSPECIFIED;
	theora_info_.quality=20;
	theora_info_.target_bitrate=0;

	theora_info_.aspect_numerator = 1;
	theora_info_.aspect_denominator = 1;

	// TODO set fps
	theora_info_.fps_numerator = 0;
	theora_info_.fps_denominator = 0;

	ctx_ = {th_encode_alloc(&theora_info_),[](th_enc_ctx* p){th_encode_free(p);}};
	if (!ctx_) {
		log[log::error] << "Failed to allocate context";
		return false;
	}
	ogg_stream_init(&ogg_state_, 5);

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
	const auto res = PLANE_DATA(frame,index).get_resolution();
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

	th_encode_ycbcr_in(ctx_.get(), buffer);

	ogg_packet packet;
	while (th_encode_packetout(ctx_.get(), 0, &packet)) {
		process_packet(packet);
	}

	return {};
}
bool TheoraEncoder::set_param(const core::Parameter& param)
{
	return base_type::set_param(param);
}

} /* namespace theora */
} /* namespace yuri */
