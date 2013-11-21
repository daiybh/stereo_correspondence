/*!
 * @file 		UVAlsaInput.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
	std::string device_name_;
	size_t capture_channels_;
};

} /* namespace uv_alsa_input */
} /* namespace yuri */
#endif /* UVALSAINPUT_H_ */
