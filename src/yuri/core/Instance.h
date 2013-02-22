/*!
 * @file 		Instance.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_
#include <yuri/core/config_common.h>
#include <yuri/exception/Exception.h>


namespace yuri {

namespace core {
typedef yuri::shared_ptr<class Instance> pInstance;

class EXPORT Instance {
public:
								Instance(std::string id, generator_t generator,
						pParameters par);
	virtual 					~Instance();
	pBasicIOThread			 	create_class(log::Log& log_, pwThreadBase parent);

	template<typename T>void 	add_parameter(std::string name,T def);
	Parameter& 					operator[] (const std::string id);
	std::string 				id;
	generator_t 				generator;
	pParameters 				params;

};

template<typename T> void Instance::add_parameter(std::string name, T val)
{
	(*params)[name]=val;
}

}

}

#endif /* INSTANCE_H_ */
