/*!
 * @file 		DummyModule.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/io/BasicIOThread.h"

namespace yuri {
namespace dummy_module {

class DummyModule: public yuri::io::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<config::Parameters> configure();
	virtual ~DummyModule();
private:
	DummyModule(log::Log &log_,io::pThreadBase parent,config::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(config::Parameter& param);
	std::string dummy_name;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
