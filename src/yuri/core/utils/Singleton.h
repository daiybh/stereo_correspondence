/*!
 * @file 		Singleton.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		8.9.2013
 * @date		19.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_
#include "yuri/core/utils/new_types.h"

namespace yuri {
namespace utils {

template<class T> EXPORT T& get_instance_helper();

#define SINGLETON_DECLARE_HELPER(S) \
namespace yuri {\
namespace utils {  \
	template<> EXPORT typename S::value_type& get_instance_helper<typename S::value_type>();\
}\
}

#define SINGLETON_DEFINE_HELPER(S) \
namespace yuri {\
namespace utils {  \
	template<> typename S::value_type& get_instance_helper<typename S::value_type>() \
	{\
		static  S::value_type instance;\
		return instance; \
	}\
}\
}

	template
	<class T
	>
class Singleton: public T {
public:
	typedef T value_type;
	EXPORT static T& get_instance() {
		return get_instance_helper<T>();
	}

private:
	Singleton();
	virtual ~Singleton() noexcept {};
private:
	Singleton(Singleton& rhs);
};




}
}


#endif /* SINGLETON_H_ */
