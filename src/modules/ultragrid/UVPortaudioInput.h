/*!
 * @file 		UVPortaudioInput.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVPORTAUDIOINPUT_H_
#define UVPORTAUDIOINPUT_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace ultragrid {

class UVPortaudioInput: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVPortaudioInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVPortaudioInput() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	int device_idx_;
	void* device_;
};

} /* namespace ultragrid */
} /* namespace yuri */
#endif /* UVPORTAUDIOINPUT_H_ */
