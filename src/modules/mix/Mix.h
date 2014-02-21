/*!
 * @file 		Mix.h
 * @author 		<Your name>
 * @date 		21.02.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef MIX_H_
#define MIX_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace mix {

class Mix: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Mix(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Mix() noexcept;
private:
	
	virtual bool step() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual	void do_connect_in(position_t, core::pPipe pipe) override;
};

} /* namespace mix */
} /* namespace yuri */
#endif /* MIX_H_ */
