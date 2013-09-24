/*
 * IOThreadGenerator.h
 *
 *  Created on: 9.9.2013
 *      Author: neneko
 */

#ifndef IOTHREADGENERATOR_H_
#define IOTHREADGENERATOR_H_
#include "yuri/core/thread/IOThread.h"
#include "yuri/core/utils/BasicGenerator.h"
#include "yuri/core/utils/Singleton.h"
#include "yuri/core/parameter/Parameters.h"
#include "yuri/exception/InitializationFailed.h"
#include <string>

namespace yuri {


template<
	class T,
	class KeyType>
class BasicIOThreadGenerator: public core::BasicGenerator<T, KeyType,
		core::Parameters,
		core::generator::DefaultErrorPolicy,
		function<shared_ptr<T> (yuri::log::Log &log, yuri::core::pwThreadBase parent, const core::Parameters& params)>>
{
public:
	BasicIOThreadGenerator(){}
};

typedef utils::Singleton<BasicIOThreadGenerator<core::IOThread, std::string>> IOThreadGenerator;

#ifdef YURI_MODULE_IN_TREE
#define REGISTER_IOTHREAD(name, type) namespace { bool reg_ ## type = yuri::IOThreadGenerator::get_instance().register_generator(name,type::generate, type::configure); }
#else
#define REGISTER_IOTHREAD(name, type) /*bool iothread_reg_ ## type = */yuri::IOThreadGenerator::get_instance().register_generator(name,type::generate, type::configure);
#endif




}



#endif /* IOTHREADGENERATOR_H_ */
