/*
 * CompressedVideoFrame.h
 *
 *  Created on: 4.10.2013
 *      Author: neneko
 */

#ifndef COMPRESSEDVIDEOFRAME_H_
#define COMPRESSEDVIDEOFRAME_H_

#include "yuri/core/frame/VideoFrame.h"
#include "yuri/core/utils/uvector.h"

namespace yuri {
namespace core {

class CompressedVideoFrame: public VideoFrame
{
public:
	typedef	uint8_t				value_type;
	typedef uvector<value_type>	vector_type;
	typedef typename vector_type::iterator
								iterator;
	typedef typename vector_type::const_iterator
								const_iterator;
	typedef typename vector_type::reference
								reference;
	typedef typename vector_type::const_reference
								const_reference;

	CompressedVideoFrame(format_t format, resolution_t resolution);
	CompressedVideoFrame(format_t format, resolution_t resolution, const uint8_t* data, size_t size);
	template<class Deleter>
	CompressedVideoFrame(format_t format, resolution_t resolution, const uint8_t* data, size_t size, Deleter deleter);
	~CompressedVideoFrame() noexcept;

	template<class... Args>
	static pFrame				create_empty(Args... args)
	{
		return make_shared<CompressedVideoFrame>(std::forward<Args>(args)...);
	}

	vector_type&	get_data() { return data_; }

	const vector_type& get_data() const { return data_; }


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

//	void						push_back(const value_type& plane);
//	void						push_back(value_type&& plane);
//	template<class... Args>
//	void						emplace_back(Args&&... args) { data_.emplace_back(std::forward<Args>(args)...); }
private:
	// TODO Implement this!
	virtual pFrame	do_get_copy() const { return pFrame(); }
	virtual size_t	do_get_size() const noexcept { return size(); };

	vector_type data_;
};

template<class Deleter>
CompressedVideoFrame::CompressedVideoFrame(format_t format, resolution_t resolution, const uint8_t* data, size_t size, Deleter deleter)
:VideoFrame(format, resolution)
{
	data_.set(data, size, deleter);
}

}
}


#endif /* COMPRESSEDVIDEOFRAME_H_ */
