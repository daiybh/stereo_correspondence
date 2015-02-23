/*
 * irange.h
 *
 *  Created on: Jan 11, 2015
 *      Author: worker
 */

#ifndef SRC_YURI_CORE_UTILS_IRANGE_H_
#define SRC_YURI_CORE_UTILS_IRANGE_H_

namespace yuri {

template<typename T>
struct irange_t {
        irange_t(T first, T end) noexcept
        		:first_(first),end_(end) {}

        struct iterator {
                iterator(T i) noexcept :i(i) {}
                iterator& operator++() noexcept {i++; return  *this; }
                iterator& operator++(int) noexcept {++i; return  *this; }
                bool operator==(const iterator& rhs) noexcept { return rhs.i == i;}
                bool operator!=(const iterator& rhs) noexcept { return rhs.i != i;}
                T operator*() noexcept { return i; }

                T i;
        };
        iterator begin() noexcept {return {first_};}
        iterator end() noexcept {return {end_};}
private:
        T first_;
        T end_;
};


template<typename T, typename T2>
typename std::enable_if<std::is_convertible<T2, T>::value, irange_t<T>>::type
irange(T first, T2 end) {
        return {first, static_cast<T>(end)};
}



}



#endif /* SRC_YURI_CORE_UTILS_IRANGE_H_ */
