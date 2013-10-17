/*!
 * @file 		UVRtpSender.h
 * @author 		<Your name>
 * @date 		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVRTPSENDER_H_
#define UVRTPSENDER_H_

#include "yuri/core/thread/IOFilter.h"
struct rtp;
struct tx;

namespace yuri {
namespace uv_rtp_sender {

class UVRtpSender: public core::IOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVRtpSender(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVRtpSender() noexcept;
private:
	
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	rtp* rtp_session_;
	tx* tx_session_;
	std::string destination_;
	uint16_t	rx_port_;
	uint16_t	tx_port_;
	int			ttl_;

};

} /* namespace uv_rtp_sender */
} /* namespace yuri */
#endif /* UVRTPSENDER_H_ */
