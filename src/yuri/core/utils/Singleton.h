/*
 * Singleton.h
 *
 *  Created on: 8.9.2013
 *      Author: neneko
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_
#include "yuri/core/utils/new_types.h"

namespace yuri {
namespace utils {
/*
template<class T>
class SingleThreaded {
protected:
	typedef T instance_type;
	struct lock_type {
		lock_type() {}
		~lock_type(){}
	};
	void lock() {}
	SingleThreaded() {}
	~SingleThreaded() {}


};

template<class T>
class MultiThreaded {
protected:
	typedef volatile T instance_type;
	struct lock_type{
		lock_type():lock_guard_(lock_){}
		~lock_type() {};
		lock_t lock_guard_;
	private:
		static mutex lock_;
	};
	MultiThreaded() {}
	~MultiThreaded() {}
};
template<class T> std::mutex MultiThreaded<T>::lock_type::lock_;

template <class T>
class NewCreationPolicy: public T {
protected:
	NewCreationPolicy() {}
	~NewCreationPolicy() {}
public:
	static NewCreationPolicy<T>* create()
	{
		return new NewCreationPolicy<T>;
	}
public:

};
*/
template
	<class T//,
//	template<class> class ThreadingPolicy = SingleThreaded,
//	template<class> class CreationPolicy = NewCreationPolicy
	>
class Singleton: public T/*, public ThreadingPolicy<T> */{
public:
	typedef T value_type;
	//typedef typename ThreadingPolicy<T>::instance_type instance_type;
	static T& get_instance()
	{
		static T 	instance_;
		return instance_;
//		if (!instance_) {
//			typename ThreadingPolicy<T>::lock_type l;
//			if (!instance_)
//				instance_ = CreationPolicy<T>::create();
//			if (!instance_) throw std::runtime_error("Failed to create an instance");
//		}
//		return *(const_cast<T*>(instance_)); // Casting away volatile for multithreaded policy
	}
private:
	Singleton();
	virtual ~Singleton() noexcept;
private:
	Singleton(Singleton& rhs);
//	static instance_type* instance_;
};


}
}


#endif /* SINGLETON_H_ */
