/*!
 * @file 		UVRtpSender.cpp
 * @author 		<Your name>
 * @date		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVRtpSender.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/ultragrid/YuriUltragrid.h"
extern "C" {
#include "transmit.h"
#include "rtp/rtp.h"
#include "video_frame.h"
}
namespace yuri {
namespace uv_rtp_sender {


IOTHREAD_GENERATOR(UVRtpSender)

MODULE_REGISTRATION_BEGIN("uv_rtp_sender")
		REGISTER_IOTHREAD("uv_rtp_sender",UVRtpSender)
MODULE_REGISTRATION_END()

core::Parameters UVRtpSender::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVRtpSender");
	return p;
}


UVRtpSender::UVRtpSender(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOFilter(log_,parent, std::string("uv_rtp_sender")),
rtp_session_(nullptr),tx_session_(nullptr)
{
	IOTHREAD_INIT(parameters)

	if (!(tx_session_ = tx_init(nullptr, 1300, TX_MEDIA_VIDEO, nullptr, nullptr))) {
		log[log::fatal] << "Failed to prepare tx session";
		throw exception::InitializationFailed("Failed to prepare tx session");
	}
	if (!(rtp_session_ = rtp_init("147.32.211.178", 2200, 2300, 200, 5000*1048675, 0, nullptr, nullptr,false))) {
		log[log::fatal] << "Failed to prepare rtp session";
		throw exception::InitializationFailed("Failed to prepare rtp session");
	}

}

UVRtpSender::~UVRtpSender() noexcept
{
//	rtp_done(rtp_session_);
}

core::pFrame UVRtpSender::do_simple_single_step(const core::pFrame& framex)
{
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(framex);
	if (frame) {
		video_frame* f = ultragrid::allocate_uv_frame(frame);
		if (f) {
			log[log::info] << "Sending rtp";
			tx_send(tx_session_, f, rtp_session_);
		}
		vf_free(f);
	}
	return {};
}
bool UVRtpSender::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_rtp_sender */
} /* namespace yuri */
