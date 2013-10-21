/*!
 * @file 		UVAlsaInput.h
 * @author 		<Your name>
 * @date 		21.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef UVALSAINPUT_H_
#define UVALSAINPUT_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace uv_alsa_input {

class UVAlsaInput: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVAlsaInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVAlsaInput() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	void* device_;
};

} /* namespace uv_alsa_input */
} /* namespace yuri */
#endif /* UVALSAINPUT_H_ */
