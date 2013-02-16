/*!
 * @file 		uvector.h
 * @author 		Zdenek Travnicek
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */


#ifndef UVECTOR_H_
#define UVECTOR_H_
#include <cstddef>
#include <algorithm>
namespace yuri {

/*!
 * Simple std::vector like class providing uninitialized storage.
 * It's @em NOT suitable for storing classes etc as it doesn't call constructors.
 */
template<typename T, bool Realloc = true>
class uvector {
public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	static const bool reallocating = Realloc;
	uvector():data_(0),size_(0),allocated_(0) {}
	uvector(size_type count):data_(new T[count]),size_(count),allocated_(0) {}
	uvector(size_type count, const T& value):data_(new T[count]),size_(count),allocated_(count) {
		std::fill(data_,data_+size_,value);
	}
	~uvector() { delete [] data_;}

	uvector(const uvector<T,Realloc>& other_):data_(0),size_(other_.size_),allocated_(other_.allocated_) {
		data_ = new T[allocated_];
		std::copy(other_.begin(),other_.end(),begin());
	}
	uvector<T,Realloc> operator=(const uvector<T,Realloc>& rhs) {
		resize(rhs.size());
		std::copy(rhs.begin(),rhs.end(),begin());
		return *this;
	}

	size_type size() const { return size_;}
	bool empty() const { return size_ == 0; }
	size_type capacity() const { return allocated_;}
	reference operator[](size_type pos) { return data_[pos]; }
	const_reference operator[](size_type pos) const { return data_[pos]; }
	reference at(size_type pos) { if (pos>=size_) throw std::out_of_range(""); return data_[pos]; }
	const_reference at(size_type pos) const { if (pos>=size_) throw std::out_of_range(""); return data_[pos]; }

	reference front() { return data_[0]; }
	const_reference front() const { return data_[0]; }
	reference back() { return data_[size_-1]; }
	const_reference back() const { return data_[size_-1]; }

	iterator begin() { assert(data_); return &data_[0]; }
	const_iterator begin() const { return &data_[0]; }
	iterator end() { return &data_[size_]; }
	const_iterator end() const { return &data_[size_]; }


	void reserve(size_type size) {reserve_impl<Realloc>(size);}


	void resize(size_type size) {
		reserve(size); size_ = size;
	}
	void clear() { size_=0;}

	iterator insert( iterator pos, const T& value);
	void insert( iterator pos, size_type count, const T& value );
	template< class InputIt >
	void insert( iterator pos, InputIt first, InputIt last) {
		size_type count = std::distance(first,last);
		reserve(size_+count);
		if (pos<end()) std::copy_backward(pos,end(),end());
		std::copy(first,last,pos);
		size_ += count;
	}
	void assign( size_type count, const T& value ) {
		size_=count; std::fill(&data_[0],data_[size_],value);}

	template< class InputIt >
	void assign( InputIt first, InputIt last ) {
		size_=std::distance(first,last);
		std::copy(first,last,&data_[0]);
	}

	void swap( uvector<T,Realloc>& other ){
		using std::swap;
		swap(allocated_,other.allocated_);
		swap(size_,other.size_);
		swap(data_,other.data_);
	}
	operator uvector<T,!Realloc>&() {
		return *(reinterpret_cast<uvector<T,!Realloc>* >(this));
	}
private:
	T*	data_;
	size_type size_;
	size_type allocated_;
	template<bool R>
	void reserve_impl(size_type size, char(*)[R]=0) {
		if (allocated_<size) {
			T* tmp = new T[size];
			if (data_) std::copy(data_,data_+size_,tmp);
			std::swap(data_,tmp);
			allocated_=size;
			delete [] tmp;
		}
	}
	template<bool R>
	void reserve_impl(size_type size, char(*)[!R]=0) {
		if (allocated_<size) {
			delete [] data_;
			data_ = new T[size];
			allocated_=size;
		}
	}
};

	template<typename T, bool ReallocL, bool ReallocR>
	void swap(yuri::uvector<T,ReallocL>& lhs, yuri::uvector<T,ReallocR>& rhs) {
		lhs.swap(rhs);
	}
}
#endif /* UVECTOR_H_ */
