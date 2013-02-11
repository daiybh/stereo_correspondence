/*
 * DummyModule.h
 *
 *  Created on: 11.2.2013
 *      Author: neneko
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/io/BasicIOThread.h"

namespace yuri {
namespace dummy_module {
using yuri::log::Log;
using yuri::config::Parameter;
using yuri::config::Parameters;
using yuri::io::pThreadBase;

class DummyModule: public yuri::io::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual ~DummyModule();
private:
	DummyModule(Log &log_,pThreadBase parent,Parameters &parameters);
	virtual bool step();
	virtual bool set_param(Parameter& param);
	std::string dummy_name;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
