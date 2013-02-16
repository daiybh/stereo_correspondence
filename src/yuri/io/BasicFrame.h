/*!
 * @file 		BasicFrame.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICFRAME_H_
#define BASICFRAME_H_

#include <vector>
#include "yuri/io/types.h"
#include <yuri/config/Config.h>
#include "pipe_types.h"
#include "yuri/io/uvector.h"
#include <algorithm>
namespace yuri {

namespace io {

typedef yuri::uvector<yuri::ubyte_t,false> plane_t;

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
	virtual yuri::size_t 		get_size();
	virtual yuri::size_t 		get_planes_count();
	virtual yuri::usize_t 		get_width();
	virtual yuri::usize_t 		get_height();
	virtual yuri::format_t 		get_format();
	virtual yuri::usize_t 		get_sample_count();
	virtual yuri::usize_t 		get_channel_count();
	virtual inline yuri::size_t get_pts() { return pts; }
	virtual inline yuri::size_t get_dts() { return dts; }
	virtual inline yuri::size_t get_duration() { return duration; }


	virtual void 				set_planes_count(yuri::size_t count);
	virtual void 				set_parameters(yuri::format_t format = YURI_FMT_NONE,
			yuri::size_t width = 0, yuri::size_t height = 0, yuri::usize_t channels=1,
			yuri::usize_t samples=1);
	virtual void 				set_time(yuri::size_t pts, size_t dts = 0,
			size_t duration = 0);

	virtual plane_t& 			get_plane(yuri::size_t index);
	virtual plane_t& 			operator[](yuri::size_t index);
	virtual void 				set_plane(yuri::size_t index, plane_t& plane) DEPRECATED;
	virtual void 				set_plane(yuri::size_t index, const plane_t& plane) DEPRECATED;
	virtual void				set_plane(yuri::size_t index, const yuri::ubyte_t *data, yuri::size_t data_size);

	virtual pBasicFrame 		get_copy();
	virtual pFrameInfo 			get_info() {return info;}
	virtual void 				set_info(pFrameInfo i) {info=i;}
protected:
	std::vector<plane_t> planes;
	yuri::size_t /*numberPlanes, */dts, pts, duration;
	yuri::format_t format;
	yuri::size_t width, height;
	yuri::usize_t samples, channels;
	pFrameInfo info;
};

}

}

#endif /* BASICFRAME_H_ */
