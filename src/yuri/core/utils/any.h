/*!
 * @file 		any.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17. 9. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_ANY_H_
#define SRC_YURI_CORE_UTILS_ANY_H_

#include <memory>
#include <type_traits>
#include "make_unique.h"
namespace yuri {
namespace core {
namespace utils {


class any {
private:
	struct data_t {
		virtual ~data_t() noexcept = default;
		virtual std::unique_ptr<data_t> copy() = 0;
	};

	template<class T>
	struct data_spec_t:public data_t {
		data_spec_t(T val):value(std::move(val)) {}
		virtual std::unique_ptr<data_t> copy() override {
			return make_unique<data_spec_t<T>>(value);
		}
		T value;
	};

	template<typename T>
	using none = typename std::enable_if<!std::is_same<any, T>::value>::type;

public:
	/*!
	 * Default constructor
	 */
	any() = default;
	/*!
	 * Copy constructor
	 * @param rhs Other instance
	 */
	any(const any& rhs)  {
		if (rhs.data_) {
			data_ = rhs.data_->copy();
		} else {
			data_.reset();
		}
	}
	/*!
	 * Move constructor
	 * @param rhs other instance
	 */
	any(any&&) noexcept = default;
	/*!
	 * Copy assignment operator
	 * @param rhs
	 * @return
	 */
	any& operator=(const any& rhs) {
		if (rhs.data_) {
			data_ = rhs.data_->copy();
		} else {
			data_.reset();
		}
		return *this;
	}
	/*!
	 * Move assignment operator
	 * @param
	 * @return
	 */
	any& operator=(any&&) noexcept = default;

	/*!
	 * Constructor from any type except any.
	 * @param val
	 */
	template<class T, typename U = typename std::decay<T>::type, typename = none<U>>
	explicit any(T&& val){
		data_ = make_unique<data_spec_t<U>>(std::forward<T>(val));
	}
	/*!
	 * Copy assignment operator from any type except any
	 * @param val
	 * @return
	 */
	template<class T, typename U = typename std::decay<T>::type, typename = none<U>>
	any& operator=(T&& val) {
		data_ = make_unique<data_spec_t<U>>(std::forward<T>(val));
		return *this;
	}

	/*!
	 * Accesses the stored value
	 * @return the stored value
	 * @throws std::bad_cast, if the contained value is of other type.
	 */
	template<class T, typename U = typename std::decay<T>::type>
	T get() const {
		auto d = dynamic_cast<data_spec_t<U>*>(data_.get());
		if (!d) throw std::bad_cast();
		return d->value;
	}
	/*!
	 * Query presence of contained value
	 * @return true, if there's a stored value
	 */
	bool empty() const { return !data_; }

	/*!
	 * Query the stored type
	 * @return true, if the stored type is the same as @em T
	 */
	template<class T, typename U = typename std::decay<T>::type>
	bool is() const{
		const auto d = dynamic_cast<data_spec_t<U>*>(data_.get());
		return d != nullptr;
	}
private:
	std::unique_ptr<data_t> data_;
};



}
}
}



#endif /* SRC_YURI_CORE_UTILS_ANY_H_ */
