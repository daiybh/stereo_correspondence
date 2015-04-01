/*!
 * @file 		managed_resource.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		1. 4. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_MANAGED_RESOURCE_H_
#define SRC_YURI_CORE_UTILS_MANAGED_RESOURCE_H_
#include <type_traits>
#include <functional>
namespace yuri {
namespace core {
namespace utils {

template<class T>
class managed_resource {
public:
	using value_type = typename std::remove_reference<T>::type;
	using ptr_type = typename std::add_pointer<value_type>::type;
	using reference_type = typename std::add_lvalue_reference<value_type>::type;
	using ptr_reference_type =  typename std::add_lvalue_reference<ptr_type>::type;
	using deleter_type = std::function<void(ptr_type)>;


	managed_resource():
		ptr_(nullptr) {}
	managed_resource(ptr_type ptr, deleter_type deleter = {}):
		ptr_(ptr),deleter_(deleter) {}

	void reset(ptr_type ptr = nullptr)
	{
		delete_impl(ptr_);
		ptr_ = ptr;
	}
	void reset(ptr_type ptr, deleter_type deleter)
	{
		reset(ptr);
		deleter_ = deleter;
	}

	ptr_type release() noexcept
	{
		return ptr_.release();
	}

	ptr_type get() const noexcept
	{
		return ptr_;
	}
	// Unsafe but usable ;)
	ptr_reference_type get_ptr_ref() noexcept
	{
		return ptr_;
	}
	ptr_type operator->() const noexcept
	{
		return ptr_;
	}

	reference_type operator*() const
	{
		return *ptr_;
	}
	explicit operator bool() const noexcept
	{
		return ptr_;
	}

	operator ptr_type() const noexcept
	{
		return get();
	}
private:
	void delete_impl(ptr_type ptr)
	{
		if (ptr_) {
			if (deleter_) deleter_(ptr);
			else delete ptr;
		}
		ptr_ = nullptr;
	}
	ptr_type ptr_;
	deleter_type deleter_;

};

}
}
}



#endif /* SRC_YURI_CORE_UTILS_MANAGED_RESOURCE_H_ */
