/*!
 * @file 		UVRtpSender.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVRtpSender.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "YuriUltragrid.h"
extern "C" {
#include "transmit.h"
#include "rtp/rtp.h"
}
#include "uv_video.h"
namespace yuri {
namespace uv_rtp_sender {


IOTHREAD_GENERATOR(UVRtpSender)


core::Parameters UVRtpSender::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVRtpSender");
	p["address"]["Target address (hostname or IP address (4 or 6))"]="127.0.0.1";
	p["rx_port"]["RX port number"]=5004;
	p["tx_port"]["TX port number"]=5004;
	p["ttl"]["TTL"]=255;
	return p;
}


UVRtpSender::UVRtpSender(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOFilter(log_,parent, std::string("uv_rtp_sender")),
rtp_session_(nullptr),tx_session_(nullptr)
{
	IOTHREAD_INIT(parameters)

	if (!(tx_session_ = tx_init(nullptr, 1500, TX_MEDIA_VIDEO, nullptr, nullptr, 0))) {
	//if (!(tx_session_ = tx_init(1300, nullptr))) {
		log[log::fatal] << "Failed to prepare tx session";
		throw exception::InitializationFailed("Failed to prepare tx session");
	}
	if (!(rtp_session_ = rtp_init(destination_.c_str(),
				rx_port_, tx_port_, ttl_,
				5000.0*1048675, 0, nullptr, nullptr,false, true))) {
		log[log::fatal] << "Failed to prepare rtp session";
		throw exception::InitializationFailed("Failed to prepare rtp session");
	}

}

UVRtpSender::~UVRtpSender() noexcept
{
//	rtp_done(rtp_session_);
}

core::pFrame UVRtpSender::do_simple_single_step(const core::pFrame& frame)
{
        if (auto videoFrame = dynamic_pointer_cast<core::VideoFrame>(frame)) {
                return do_special_simple_single_step(videoFrame);
        } else if (auto audioFrame = dynamic_pointer_cast<core::AudioFrame>(frame)) {
                return do_special_simple_single_step(audioFrame);
        } else return {};
}
core::pFrame UVRtpSender::do_special_simple_single_step(const core::pVideoFrame& frame)
{
	auto f = ultragrid::allocate_uv_frame(frame);
	if (f) {
		tx_send(tx_session_, f.get(), rtp_session_);
//		vf_free(f);
	}
	return {};
}
core::pFrame UVRtpSender::do_special_simple_single_step(const core::pAudioFrame& frame)
{
	auto f = ultragrid::allocate_uv_frame(frame);
	if (f) {
                audio_tx_send(tx_session_, rtp_session_, f.get());
	}
	return {};
}
bool UVRtpSender::set_param(const core::Parameter& param)
{
	if (param.get_name() == "address") {
		destination_=param.get<std::string>();
	} else if (param.get_name() == "rx_port") {
		rx_port_=param.get<uint16_t>();
		if (rx_port_%2) rx_port_++;
	} else if (param.get_name() == "tx_port") {
		tx_port_=param.get<uint16_t>();
		if (tx_port_%2) tx_port_++;
	} else if (param.get_name() == "ttl") {
		ttl_=param.get<int>();
	} else return core::IOFilter::set_param(param);
	return true;
}

} /* namespace uv_rtp_sender */
} /* namespace yuri */
