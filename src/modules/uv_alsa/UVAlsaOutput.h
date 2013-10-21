/*!
 * @file 		UVAlsaOutput.h
 * @author 		<Your name>
 * @date 		21.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVALSAOUTPUT_H_
#define UVALSAOUTPUT_H_


#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawAudioFrame.h"

namespace yuri {
namespace uv_alsa_output {

class UVAlsaOutput: public core::SpecializedIOFilter<core::RawAudioFrame>
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVAlsaOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVAlsaOutput() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawAudioFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);\

	void* device_;
};

} /* namespace uv_alsa_output */
} /* namespace yuri */
#endif /* UVALSAOUTPUT_H_ */
