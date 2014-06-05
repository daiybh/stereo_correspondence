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
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "YuriUltragrid.h"
extern "C" {
#include "debug.h"
#include "pdb.h"
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/video_decoders.h"
#include "tfrc.h"
#include "tv.h"
}
#include "uv_video.h"

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
        m_participants = pdb_init();
	if (!(rtp_session_ = rtp_init(destination_.c_str(),
				rx_port_, tx_port_, ttl_,
				5000*1048675, 0, rtp_recv_callback, (uint8_t *) m_participants, false, true))) {
		log[log::fatal] << "Failed to prepare rtp session";
		throw exception::InitializationFailed("Failed to prepare rtp session");
	}

        rtp_set_option(rtp_session_, RTP_OPT_WEAK_VALIDATION, TRUE);
        rtp_set_sdes(rtp_session_, rtp_my_ssrc(rtp_session_),
                        RTCP_SDES_TOOL,
                        PACKAGE_STRING, strlen(PACKAGE_STRING));

        pdb_add(m_participants, rtp_my_ssrc(rtp_session_));

        gettimeofday(&m_start_time, NULL);
}

UVRtpReceiver::~UVRtpReceiver() noexcept
{
}

void UVRtpReceiver::run()
{
        while (still_running()) {
                struct timeval curr_time;
                struct timeval timeout;
                uint32_t ts;
                int fr = 1;

                /* Housekeeping and RTCP... */
                gettimeofday(&curr_time, NULL);
                ts = tv_diff(curr_time, m_start_time) * 90000;
                rtp_update(rtp_session_, curr_time);
                rtp_send_ctrl(rtp_session_, ts, 0, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&curr_time, NULL);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                //timeout.tv_usec = 999999 / 59.94;
                timeout.tv_usec = 10000;
                int ret = rtp_recv_r(rtp_session_, &timeout, ts);

                // timeout
                if (ret == FALSE) {
                        // processing is needed here in case we are not receiving any data
                        //printf("Failed to receive data\n");
                }

                /* Decode and render for each participant in the conference... */
                pdb_iter_t it;
                struct pdb_e *cp = pdb_iter_init(m_participants, &it);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               curr_time));
                        }

                        yuri_decoder_data decoder_data;
                        decoder_data.log = (void *) &log;
                        /// @todo remove ugly taking of raw pointers
                        auto lambda = [](struct video_desc *desc, size_t size, char **data, void *log) {
                                core::pFrame frame = ultragrid::create_yuri_from_uv_desc(desc, size, (log::Log &)*log);
                                auto raw = dynamic_pointer_cast<core::RawVideoFrame>(frame);
                                if (raw) {
                                        *data = reinterpret_cast<char *>(PLANE_RAW_DATA(raw, 0));
                                } else {
                                                auto compressed = dynamic_pointer_cast<core::CompressedVideoFrame>(frame);
                                        if (compressed) {
                                                *data = reinterpret_cast<char *>(compressed->begin());
                                        }
                                }
                                return frame;
                        };
                        decoder_data.create_yuri_frame = static_cast<core::pFrame (*)(struct video_desc *, size_t, char **, void *)>(lambda);

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_yuri_frame, (void *) &decoder_data)) {
                                gettimeofday(&curr_time, NULL);
                                fr = 1;
                                push_frame(0, decoder_data.yuri_frame);
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(&it);
                }
                pdb_iter_done(&it);
        }
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
