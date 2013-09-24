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

	static pRawVideoFrame create_empty(format_t, resolution_t, bool fixed = false, interlace_t interlace = interlace_t::progressive, field_order_t filed_order = field_order_t::none);

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

}
}



#endif /* RAWVIDEOFRAME_H_ */
