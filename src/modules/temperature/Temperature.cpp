/*!
 * @file 		Temperature.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Temperature.h"
#include "yuri/config/RegisteredClass.h"
#include <limits>
namespace yuri {
namespace temperature {

REGISTER("temperature",Temperature)

IO_THREAD_GENERATOR(Temperature)

shared_ptr<config::Parameters> Temperature::configure()
{
	shared_ptr<config::Parameters> p = io::BasicIOThread::configure();
	p->set_description("Temperature.");
	p->set_max_pipes(1,1);
	return p;
}


Temperature::Temperature(log::Log &log_,io::pThreadBase parent,config::Parameters &parameters):
io::BasicIOThread(log_,parent,1,1,std::string("temperature"))
{
	IO_THREAD_INIT("Temperature")
}

Temperature::~Temperature()
{
}
namespace {
inline double r(double val) {
	//return val<0.8?0.0:val<0.99?val*5.0-4.0:0.0;
	// 0 - 0.35 -> 0.0
	// 0.35 - 0.66 -> going up (0.0 - 1.0)
	// 0.65 - 0.9 -> 1.0
	// 0.9 - 1.0 going down (1.0 - 0.5)
	return val<0.35?0.0:val<0.65?val*4.0-1.4:val<0.9?1.0:5.5-val*5.0;
}
inline double g(double val) {
	//return (val<0.4)?0:val<0.6?val*5.0-2.0:val<0.8?4.0-5*val:0.0;
	//return (val<0.4||val>0.8)?0.0:val*2.5-1.0;
	//return (val<0.2)?0:val<0.4?val*5.0-1.0:val<0.8?1.0:5.0-5.0*val;


	// 0.0 - 0.1 -> 0.0
	// 0.1 - 0.35 -> going up (0.0 - 1.0)
	// 0.35 - 0.65 -> 1.0
	// 0.65 - 0.9 -> going down (1.0 - 0.0)
	return val<0.1?0.0:val<0.35?val*4.0-0.4:val<0.65?1.0:val<0.9?3.6-val*4.0:0.0;
}
inline double b(double val) {
	//return val>0.6?0.0:val*2.0;
	return val<0.2?val*5.0:val<0.4?2.0-5.0*val:0.0;

	// 0.0 - 0.1 -> going up (0.5->1.0)
	// 0.1 - 0.35 -> 1.0
	// 0.35 - 0.65 -> going down (1.0-0.0)
	// 0.65 - 1.0 -> 0.0
	return val<0.1?val*5.0+0.5:val<0.35?1.0:val<0.65?(6.5)/3.0-(10.0/3.0)*val:0.0;
}
template<typename T>
io::pBasicFrame colorize(const io::pBasicFrame& frame)
{
	const size_t size = PLANE_SIZE(frame,0)/sizeof(T);
	T* data = reinterpret_cast<T*>(PLANE_RAW_DATA(frame,0));
	T lower_bound = std::numeric_limits<T>::min();
	T upper_bound = std::numeric_limits<T>::max();
	const size_t width = frame->get_width();
	const size_t height = frame->get_height();
	assert(size==width*height);
	io::pBasicFrame output = io::BasicIOThread::allocate_empty_frame(YURI_FMT_RGB24, width, height);
	io::pFrameInfo fi = frame->get_info();
	if (fi) {
		if (fi->min_value) lower_bound = fi->min_value;
		if (fi->max_value) upper_bound = fi->max_value;
	}
	ubyte_t* out = PLANE_RAW_DATA(output,0);
	for (size_t i=0;i<size;++i) {
		T& pix = *data++;
		double val = 1.0 - static_cast<double>(pix-lower_bound)/(upper_bound-lower_bound);
		if (val==1.0) {*out++=0;*out++=0;*out++=0;}
		else {
			*out++=static_cast<ubyte_t>(255*r(val));
			*out++=static_cast<ubyte_t>(255*g(val));
			*out++=static_cast<ubyte_t>(255*b(val));
		}
	}
	return output;
}
}

bool Temperature::step()
{
	io::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		yuri::format_t fmt = frame->get_format();
		io::pBasicFrame output;
		switch(fmt) {
			case YURI_FMT_RED8:
			case YURI_FMT_GREEN8:
			case YURI_FMT_BLUE8:
			case YURI_FMT_Y8:
			case YURI_FMT_U8:
			case YURI_FMT_V8:output = output = colorize<yuri::ubyte_t>(frame); break;
			case YURI_FMT_DEPTH8:
			case YURI_FMT_RED16:
			case YURI_FMT_GREEN16:
			case YURI_FMT_BLUE16:
			case YURI_FMT_Y16:
			case YURI_FMT_U16:
			case YURI_FMT_V16:
			case YURI_FMT_DEPTH16:output = colorize<yuri::ushort_t>(frame); break;
			default: output = frame; break;
		}
		if (output) push_raw_video_frame(0, output);
	}
	return true;
}
bool Temperature::set_param(config::Parameter& param)
{
	return io::BasicIOThread::set_param(param);
	//return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
