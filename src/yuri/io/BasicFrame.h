/*
 * BasicFrame.h
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#ifndef BASICFRAME_H_
#define BASICFRAME_H_

#include <vector>
#include "yuri/io/types.h"
#include <yuri/config/Config.h>
#include "pipe_types.h"

namespace yuri {

namespace io {

template<typename T> struct Plane
{
	Plane():data(),size(0) {}
	Plane(shared_array<T> data, yuri::size_t size):data(data),size(size) {}
	shared_array<T> data;
	yuri::size_t size;
	inline Plane &set(shared_array<T> data0, yuri::size_t size0)
			{data = data0; size = size0; return *this;}
	inline T& operator[](yuri::size_t index) {return data[index];}
	virtual inline yuri::size_t get_size() { return size * sizeof(T); }
	shared_ptr<Plane<T> > virtual get_copy() {
		shared_array<T> datatmp(new T[size]);
		memcpy(datatmp.get(),data.get(),size*sizeof(T));
		shared_ptr<Plane<T> > planetmp(new Plane(datatmp,size));
		return planetmp;
	}
};

typedef yuri::shared_ptr<struct FrameInfo> pFrameInfo;
struct EXPORT FrameInfo {
	virtual ~FrameInfo() {}
	std::string format;
	yuri::size_t value;
	yuri::size_t scale;
};

typedef yuri::shared_ptr<class BasicFrame> pBasicFrame;
class EXPORT BasicFrame {
public:
	BasicFrame(yuri::size_t planes = 1);
	virtual ~BasicFrame();
	virtual yuri::size_t get_size();
	virtual void set_planes_count(yuri::size_t count);
	virtual void set_parameters(yuri::format_t format = YURI_FMT_NONE, yuri::size_t width = 0, yuri::size_t height = 0, yuri::usize_t channels=1, yuri::usize_t samples=1);
	virtual void set_time(yuri::size_t pts, size_t dts = 0, size_t duration = 0);
	virtual  yuri::size_t get_planes_count();
	virtual  yuri::usize_t get_width();
	virtual  yuri::usize_t get_height();
	virtual  yuri::format_t get_format();
	virtual  yuri::usize_t get_sample_count();
	virtual  yuri::usize_t get_channel_count();
	virtual inline yuri::size_t get_pts() { return pts; }
	virtual inline yuri::size_t get_dts() { return dts; }
	virtual inline yuri::size_t get_duration() { return duration; }


	virtual Plane<yuri::ubyte_t>& operator[](yuri::size_t index);
	virtual void set_plane(yuri::size_t index, shared_ptr<Plane<yuri::ubyte_t> > plane);

	virtual pBasicFrame get_copy();
	virtual pFrameInfo get_info() {return info;}
	virtual void set_info(pFrameInfo i) {info=i;}
protected:
	std::vector<shared_ptr<Plane<yuri::ubyte_t> > > planes;
	yuri::size_t numberPlanes, dts, pts, duration;
	yuri::format_t format;
	yuri::size_t width, height;
	yuri::usize_t samples, channels;
	pFrameInfo info;
};

}

}

#endif /* BASICFRAME_H_ */
