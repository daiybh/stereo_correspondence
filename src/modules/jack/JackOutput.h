/*!
 * @file 		JackOutput.h
 * @author 		<Your name>
 * @date 		19.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef JACKOUTPUT_H_
#define JACKOUTPUT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include <jack/jack.h>

namespace yuri {
namespace jack_output {

class JackOutput: public core::SpecializedIOFilter<core::RawAudioFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawAudioFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	JackOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~JackOutput() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawAudioFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	jack_client_t* handle_;
	jack_port_t * port_;
	std::string client_name_;
	std::string port_name_;



};

} /* namespace jack_output */
} /* namespace yuri */
#endif /* JACKOUTPUT_H_ */
