/*!
 * @file 		Instance.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Instance.h"

namespace yuri {

namespace core {

Instance::Instance(std::string id, generator_t generator, shared_ptr<Parameters> par):id(id),
		generator(generator)
{
	if (par.get()) params.reset(new Parameters(*par));
	else params.reset(new Parameters());

}

Instance::~Instance() {

}

pBasicIOThread Instance::create_class(yuri::log::Log& log_, pwThreadBase parent)
{
	if (!generator) throw Exception("Tried to create class without an generator!");
	if (!params.get()) params.reset(new Parameters());
	pBasicIOThread thread(generator(log_,parent,*params));
	return thread;
}

Parameter& Instance::operator[] (const std::string id)
{
	return (*params)[id];
}

}

}
