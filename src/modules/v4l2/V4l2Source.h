/*
 * V4l2Source.h
 *
 *  Created on: May 17, 2009
 *      Author: neneko
 */

#ifndef V4L2SOURCE_H_
#define V4L2SOURCE_H_

#include "yuri/io/BasicIOThread.h"

#include <iostream>
#include <linux/videodev2.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <string>
#include <errno.h>
#include <sys/mman.h>
#include <malloc.h>
#include <yuri/exception/Exception.h>
#include <yuri/config/Config.h>
#include <yuri/config/Parameters.h>

namespace yuri {

namespace io {
using yuri::log::Log;
using yuri::exception::Exception;
using namespace yuri::config;
using namespace std;

#ifndef V4L2_CID_ILLUMINATORS_1
#define V4L2_CID_ILLUMINATORS_1			(V4L2_CID_BASE+37)
#endif

class V4l2Source: public BasicIOThread {
public:
	/** Structure to hold buffer informations */
	struct buffer_t {
	        void *                  start;
	        yuri::size_t            length;
	};
	/** Methods to read from v4l2 devices*/
	enum methods {
		    /** No known way how to read from device*/
			METHOD_NONE,
			/** Use mmap to map buffers */
			METHOD_MMAP,
			/** Use user specified buffers */
			METHOD_USER,
			/** Use direct read from the device file */
			METHOD_READ
		};

	virtual ~V4l2Source();
	virtual void run();
	static shared_ptr<BasicIOThread>  generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
/*
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getImageSize() { return imagesize; }
	int getPixelFormat() { log[info] << pixelformat << std::endl;return pixelformat; }*/
	/** Converts yuri::format_t to v4l2 format
	 * \param fmt V4l2 pixel format
	 * \return yuri::format_t for the specified format.
	 */
	static yuri::uint_t yuri_format_to_v4l2(yuri::format_t fmt) throw (Exception);
	/** Converts v4l2 format to yuri::format_t
	 * \param fmt Pixel format as yuri::format_t
	 * \return v4l2 pixel format for the specified format.
	 */
	static yuri::format_t v4l2_format_to_yuri(yuri::uint_t fmt) throw (Exception);
	virtual bool set_param(Parameter &param);
protected:
	V4l2Source(Log &log_,pThreadBase parent,Parameters &parameters)
			IO_THREAD_CONSTRUCTOR;
	bool init_mmap();
	bool init_user();
	bool init_read();
	bool start_capture();
	bool stop_capture();
	bool read_frame();
	static int xioctl(int fd, unsigned long int request, void *arg);
	bool prepare_frame(yuri::ubyte_t *data, yuri::size_t size);

	bool open_file() throw(Exception);
	bool query_capabilities() throw(Exception);
	bool enum_inputs() throw(Exception);
	bool set_input() throw(Exception);
	bool set_cropping() throw (Exception);
	bool enum_formats() throw (Exception);
	bool set_format() throw (Exception);
	bool enum_frame_intervals() throw (Exception);
	bool set_frame_params() throw (Exception);
	bool initialize_capture() throw(Exception);
	bool enable_iluminator() throw(Exception);

	std::string filename;
	int fd;
	v4l2_capability cap;
	v4l2_format fmt;
	yuri::size_t width,height,imagesize;
	uint pixelformat;
	methods method;
	bool illumination;
	buffer_t *buffers;
	yuri::ushort_t no_buffers;
	yuri::ushort_t input_number;
	yuri::size_t buffer_free;
	shared_ptr<BasicFrame> output_frame;
	bool combine_frames;
	vector<yuri::uint_t> supported_formats;
	yuri::size_t number_of_inputs;
	static map<yuri::format_t,yuri::uint_t> formats_map;
	static map<string, yuri::uint_t> special_formats;
	yuri::size_t fps;
	yuri::size_t frame_duration;
};

}

}
#endif /* V4L2SOURCE_H_ */
