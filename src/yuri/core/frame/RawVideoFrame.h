/*
 * RawVideoFrame.h
 *
 *  Created on: 30.7.2013
 *      Author: neneko
 */

#ifndef RAWVIDEOFRAME_H_
#define RAWVIDEOFRAME_H_
#include "VideoFrame.h"
#include "Plane.h"
#include "raw_frame_params.h"
#include <vector>
namespace yuri {
namespace core {
class RawVideoFrame;
typedef shared_ptr<RawVideoFrame> pRawVideoFrame;
#define PLANE_DATA(pframe, idx) (*pframe)[idx]
#define PLANE_RAW_DATA(pframe, idx) (*pframe)[idx].data()
#define PLANE_SIZE(pframe, idx) (*pframe)[idx].size()

class RawVideoFrame: public VideoFrame
{
public:
	typedef	Plane				value_type;
	typedef std::vector<value_type>
								vector_type;
	typedef typename vector_type::iterator
								iterator;
	typedef typename vector_type::const_iterator
								const_iterator;
	typedef typename vector_type::reference
								reference;
	typedef typename vector_type::const_reference
								const_reference;

	static pRawVideoFrame create_empty(format_t, resolution_t, bool fixed = false, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);
	static pRawVideoFrame create_empty(format_t, resolution_t, const uint8_t* data, size_t size, bool fixed= false, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);
	template<class Iter>
	static pRawVideoFrame create_empty(format_t frame, resolution_t resolution, Iter start, Iter end, bool fixed = false, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);
	template<class Deleter>
	static pRawVideoFrame create_empty(format_t frame, resolution_t resolution, const uint8_t* data, size_t size, Deleter deleter, interlace_t interlace = interlace_t::progressive, field_order_t field_order = field_order_t::none);


	RawVideoFrame(format_t format, resolution_t resolution, size_t plane_count = 1);
	virtual ~RawVideoFrame() noexcept;

//	Plane&	operator[](index_t index);

	size_t			get_planes_count() const { return planes_.size(); }
	void			set_planes_count(index_t count);


	iterator					begin() {return planes_.begin();}
	iterator					end() {return planes_.end();}
	const_iterator				begin() const {return planes_.begin();}
	const_iterator				end() const {return planes_.end();}
	const_iterator				cbegin() const {return planes_.cbegin();}
	const_iterator				cend() const {return planes_.cend();}
	iterator					data() { return begin(); }
	const_iterator				data() const { return begin(); }
	reference					operator[](index_t index) { return planes_[index]; }
	const_reference				operator[](index_t index) const { return planes_[index]; }
	size_t						size() const { return planes_.size(); }
	void						push_back(const Plane& plane);
	void						push_back(Plane&& plane);
	template<class... Args>
	void						emplace_back(Args&&... args) { planes_.emplace_back(std::forward<Args>(args)...); }

private:
	/*!
	 * Implementation of copy, should be implemented in node classes only.
	 * @return Copy of current frame
	 */
	virtual pFrame	do_get_copy() const;
	/*!
	 * Implementation od get_size() method
	 * @return Size of current frame
	 */
	virtual size_t	do_get_size() const noexcept;

	static std::pair<size_t, size_t>	get_plane_params(const raw_format::raw_format_t& info, size_t plane, resolution_t resolution);
	static std::pair<size_t, size_t>	get_plane_params(const raw_format::plane_info_t& info, resolution_t resolution);
protected:
	/*!
	 * Copies parameters from the frame to other frame.
	 * The intention is to be used in get_copy() method.
	 * @param frame to copy parameters to
	 */
	virtual void	copy_parameters(Frame&) const;
private:
	vector_type		planes_;

};


template<class Iter>
pRawVideoFrame RawVideoFrame::create_empty(format_t format, resolution_t resolution, Iter start, Iter end, bool fixed, interlace_t interlace, field_order_t field_order)
{
	pRawVideoFrame frame = create_empty(format, resolution, fixed, interlace, field_order);
	if (frame) {
		if (std::distance(start,end) > PLANE_SIZE(frame,0)) end = start+PLANE_SIZE(frame,0);
		std::copy(start, end, PLANE_DATA(frame,0).begin());
	}
	return frame;
}

template<class Deleter>
pRawVideoFrame RawVideoFrame::create_empty(format_t format, resolution_t resolution, const uint8_t* data, size_t size, Deleter deleter, interlace_t interlace, field_order_t field_order)
{
	pRawVideoFrame frame;
	try {
			const auto& info = raw_format::get_format_info(format);
			size_t line_size, frame_size;
			std::tie(line_size, frame_size) = get_plane_params(info, 0, resolution);
//			assert(size>= frame_size);
			// Creating with 0 planes and them emplacing planes into it.
			pRawVideoFrame frame = make_shared<RawVideoFrame>(format, resolution, 0);
			frame->set_interlacing(interlace);
			frame->set_field_order(field_order);

			frame->emplace_back(data, size, resolution, line_size, deleter);
	}
	catch(std::runtime_error&) {frame.reset();}
	return frame;
}

}
}



#endif /* RAWVIDEOFRAME_H_ */
