/*!
 * @file 		Dup.h
 * @author 		Zdenek Travnicek
 * @date 		23.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DUP_H_
#define DUP_H_

#include "yuri/io/BasicIOThread.h"
#include <yuri/config/Config.h>
#include <yuri/config/Parameters.h>
#include <yuri/config/RegisteredClass.h>
namespace yuri {

namespace io {
using yuri::log::Log;
using namespace yuri::config;
class Dup: public BasicIOThread {
public:
	virtual ~Dup();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();

	virtual void connect_out(int index,shared_ptr<BasicPipe> pipe);
	virtual bool step();
	virtual bool set_param(Parameter &parameter);
protected:
	Dup(Log &log_,pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	bool hard_dup;
};

}

}

#endif /* DUP_H_ */
