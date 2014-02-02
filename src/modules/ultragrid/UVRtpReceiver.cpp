/*!
 * @file 		UVRtpReceiver.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVRtpReceiver.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/ultragrid/YuriUltragrid.h"
extern "C" {
#include "transmit.h"
#include "rtp/rtp.h"
#include "video_frame.h"
}


namespace yuri {
namespace uv_rtp_receiver {


IOTHREAD_GENERATOR(UVRtpReceiver)

core::Parameters UVRtpReceiver::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVRtpReceiver");
	p["address"]["Target address (hostname or IP address (4 or 6))"]="127.0.0.1";
	p["rx_port"]["RX port number"]=5004;
	p["tx_port"]["TX port number"]=5004;
	p["ttl"]["TTL"]=255;
	return p;
}


UVRtpReceiver::UVRtpReceiver(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,1,std::string("uv_rtp_receiver"))
{
	IOTHREAD_INIT(parameters)

//	if (!(rx_session_ = rx_init(nullptr, 1300, TX_MEDIA_VIDEO, nullptr, nullptr))) {
//		log[log::fatal] << "Failed to prepare tx session";
//		throw exception::InitializationFailed("Failed to prepare tx session");
//	}
	if (!(rtp_session_ = rtp_init(destination_.c_str(),
				rx_port_, tx_port_, ttl_,
				5000*1048675, 0, nullptr, nullptr,false))) {
		log[log::fatal] << "Failed to prepare rtp session";
		throw exception::InitializationFailed("Failed to prepare rtp session");
	}

}

UVRtpReceiver::~UVRtpReceiver() noexcept
{
}

void UVRtpReceiver::run()
{

}
bool UVRtpReceiver::set_param(const core::Parameter& param)
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
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace uv_rtp_receiver */
} /* namespace yuri */
