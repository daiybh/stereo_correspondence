/*!
 * @file 		DummyModule.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/core/IOThread.h"

namespace yuri {
namespace dummy_module {

class DummyModule: public core::IOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~DummyModule();
private:
	DummyModule(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	std::string dummy_name;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
