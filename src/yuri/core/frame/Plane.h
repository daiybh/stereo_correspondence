/*
 * Plane.h
 *
 *  Created on: 31.7.2013
 *      Author: neneko
 */

#ifndef PLANE_H_
#define PLANE_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/uvector.h"

namespace yuri {
namespace core {

template<typename T>
class GenericPlane {
public:
	typedef	T					value_type;
	typedef uvector<value_type>	vector_type;
	typedef typename vector_type::iterator
								iterator;
	typedef typename vector_type::const_iterator
								const_iterator;
	typedef typename vector_type::reference
								reference;
	typedef typename vector_type::const_reference
							const_reference;

	GenericPlane(size_t size, resolution_t resolution, dimension_t line_size)
		:resolution_(resolution),line_size_(line_size),data_(size) {}
	GenericPlane(vector_type&& data, resolution_t resolution, dimension_t line_size)
			:resolution_(resolution),line_size_(line_size),data_(std::move(data)) {}
	GenericPlane(const GenericPlane& rhs):resolution_(rhs.resolution_),line_size_(rhs.line_size_),data_(rhs.data_)
	{ }
	GenericPlane(GenericPlane&& rhs) noexcept:resolution_(rhs.resolution_),line_size_(rhs.line_size_),data_(std::move(rhs.data_))
	{}
	template<class Deleter>
	GenericPlane(const T* data, size_t size, resolution_t resolution, dimension_t line_size, Deleter deleter);
	GenericPlane& operator=(const GenericPlane& rhs) {
		resolution_ 	= rhs.resolution_;
		line_size_ 		= rhs.line_size_;
		data_.resize(rhs.data_.size());
		std::copy(rhs.data_.begin(), rhs.data_.end(), data_.begin());
		return *this;
	}
	GenericPlane& operator=(GenericPlane&& rhs) {
		resolution_ 	= rhs.resolution_;
		line_size_ 		= rhs.line_size_;
		data_.swap(rhs.data_);
		return *this;
	}
	template<class Deleter>
	void set_data(const T* data, size_t size, Deleter deleter);

	iterator					begin() {return data_.begin();}
	iterator					end() {return data_.end();}
	const_iterator				begin() const {return data_.begin();}
	const_iterator				end() const {return data_.end();}
	const_iterator				cbegin() const {return data_.cbegin();}
	const_iterator				cend() const {return data_.cend();}
	iterator					data() { return begin(); }
	const_iterator				data() const { return begin(); }
	reference					operator[](index_t index) { return data_[index]; }
	const_reference				operator[](index_t index) const { return data_[index]; }
	size_t						size() const { return data_.size(); }

	dimension_t					get_line_size() const { return line_size_; }
	resolution_t				get_resolution() const { return resolution_; }
	size_t						get_size() const { return size() * sizeof(value_type);}
private:
	resolution_t				resolution_;
	dimension_t					line_size_;
	vector_type					data_;
};
template<typename T>
template<class Deleter>
GenericPlane<T>::GenericPlane(const T* data, size_t size, resolution_t resolution, dimension_t line_size, Deleter deleter)
:resolution_(resolution),line_size_(line_size)
{
	data_.set(data, size, deleter);
}

template<typename T>
template<class Deleter>
void GenericPlane<T>::set_data(const T* data, size_t size, Deleter deleter)
{
	data_.set(data, size, deleter);
}

typedef GenericPlane<uint8_t>	Plane;



}
}


#endif /* PLANE_H_ */
