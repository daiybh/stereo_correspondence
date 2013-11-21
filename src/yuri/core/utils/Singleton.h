/*!
 * @file 		Singleton.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		8.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_
#include "yuri/core/utils/new_types.h"

namespace yuri {
namespace utils {

template
	<class T
	>
class Singleton: public T {
public:
	typedef T value_type;
	static T& get_instance()
	{
		static T 	instance_;
		return instance_;
	}
private:
	Singleton();
	virtual ~Singleton() noexcept;
private:
	Singleton(Singleton& rhs);
};


}
}


#endif /* SINGLETON_H_ */
