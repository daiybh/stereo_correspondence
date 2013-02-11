/*
 * Instance.h
 *
 *  Created on: Jul 24, 2010
 *      Author: neneko
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_
#include <yuri/config/config_common.h>
#include <yuri/exception/Exception.h>


namespace yuri {

namespace config {
using namespace yuri::io;
using namespace yuri::exception;
using namespace yuri::log;
using namespace yuri::threads;
class Instance {
public:
	Instance(std::string id, generator_t generator, shared_ptr<Parameters> par);
	virtual ~Instance();
	shared_ptr<BasicIOThread> create_class(Log& log_,pThreadBase parent) throw(Exception);

	template<typename T>void add_parameter(std::string name,T def);
	Parameter& operator[] (const std::string id);
	std::string id;
	generator_t generator;
	shared_ptr<Parameters> params;

};

template<typename T> void Instance::add_parameter(std::string name, T val)
{
	(*params)[name]=val;
}

}

}

#endif /* INSTANCE_H_ */
