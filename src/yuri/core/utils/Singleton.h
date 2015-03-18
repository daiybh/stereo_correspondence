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


#if defined(YURI_WIN) || defined(YURI_CYGWIN)
	template
		<class T
		>
	class SingletonHelper {
	public:
		virtual T& get_instance() = 0;
	};

	template<class T> EXPORT SingletonHelper<T>& get_instance_helper();
#ifdef yuri2_8_core_EXPORTS
	template<class T>
	class SingletonHelperImpl: public SingletonHelper<T>{
	public:
		virtual T& get_instance() override {
			static T instance_;
			return instance_;
		}
	};
	template<class T> EXPORT SingletonHelper<T>& get_instance_helper()
	{
		static SingletonHelperImpl<T> sing;
		return sing;
	}


#endif
#endif

	template
	<class T
	>
class Singleton: public T {
public:
	typedef T value_type;
#if !defined(YURI_WIN) & !defined(YURI_CYGWIN)
	static T& get_instance()
	{
		static T 	instance_;
		return instance_;
	}
#else
	
//#if defined(yuri2_8_core_EXPORTS)
	/*static T& instance()
	{
		static T 	instance_;
		return instance_;
	}
*/
	EXPORT static T& get_instance() {
		return get_instance_helper<T>().get_instance();
	}
/*#else
	EXPORT static T& get_instance();
	
#endif*/
#endif

private:
	Singleton();
	virtual ~Singleton() noexcept {};
private:
	Singleton(Singleton& rhs);
};




}
}


#endif /* SINGLETON_H_ */
