/*!
 * @file 		UVRtpReceiver.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVRTPRECEIVER_H_
#define UVRTPRECEIVER_H_

#include "yuri/core/thread/IOThread.h"
#ifdef YURI_CYGWIN
#include <sys/time.h>
#endif
struct pdb;
struct rtp;
//struct rx;
namespace yuri {
namespace uv_rtp_receiver {


class UVRtpReceiver: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVRtpReceiver(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVRtpReceiver() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	rtp* rtp_session_;
        struct pdb *m_participants;
//	rx* rx_session_;
        struct timeval m_start_time;

	std::string destination_;
	uint16_t	rx_port_;
	uint16_t	tx_port_;
	int			ttl_;

};

} /* namespace uv_rtp_receiver */
} /* namespace yuri */
#endif /* UVRTPRECEIVER_H_ */
