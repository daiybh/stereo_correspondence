/*
 * array_range.h
 *
 *  Created on: 18.10.2013
 *      Author: neneko
 */

#ifndef ARRAY_RANGE_H_
#define ARRAY_RANGE_H_

namespace yuri {

/*!
 * Wrapper for dynamic arrays (or parts of them) to make them usable in range based for.
 * Intended usage is:
 * int* arr = new int[size];
 * for(auto& i: array_range<int>(arr, arr+size)) {
 * 		i=1;
 * }
 * or
 * for(auto& i: array_range<int>(arr, size)) {
 * 		i=1;
 * }
 */
template<class T>
class array_range {
public:
	using pointer_type = T*;
	using const_pointer_type = const T*;

	/*!
	 * @param first pointer to begining of the range
	 * @param last pointer to the end of the range
	 */
	explicit array_range(pointer_type first, pointer_type last):
		first(first),last(last){}

	/*!
	 * @param first pointer to beginning of the range
	 * @param size length of the range
	 */
	template<typename INT, typename dummy = typename std::enable_if<std::is_integral<INT>::value>::type>
	array_range(pointer_type first, INT size):
		first(first),last(first+size){}

	pointer_type begin() { return first; }
	const_pointer_type begin() const { return first; }
	pointer_type end() { return last; }
	const_pointer_type end() const { return last; }
private:
	pointer_type first;
	pointer_type last;
};


}


#endif /* ARRAY_RANGE_H_ */
