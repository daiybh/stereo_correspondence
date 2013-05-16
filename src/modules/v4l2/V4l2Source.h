/*!
 * @file 		V4l2Source.h
 * @author 		Zdenek Travnicek
 * @date 		17.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef V4L2SOURCE_H_
#define V4L2SOURCE_H_

#include "yuri/core/BasicIOThread.h"

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

namespace yuri {

namespace io {

#ifndef V4L2_CID_ILLUMINATORS_1
#define V4L2_CID_ILLUMINATORS_1			(V4L2_CID_BASE+37)
#endif

class V4l2Source: public core::BasicIOThread {
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
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
/*
	int getWidth() { return width; }
	int getHeight() { return height; }
	int getImageSize() { return imagesize; }
	int getPixelFormat() { log[info] << pixelformat << std::endl;return pixelformat; }*/
	/** Converts yuri::format_t to v4l2 format
	 * \param fmt V4l2 pixel format
	 * \return yuri::format_t for the specified format.
	 */
	static yuri::uint_t yuri_format_to_v4l2(yuri::format_t fmt);
	/** Converts v4l2 format to yuri::format_t
	 * \param fmt Pixel format as yuri::format_t
	 * \return v4l2 pixel format for the specified format.
	 */
	static yuri::format_t v4l2_format_to_yuri(yuri::uint_t fmt);
	virtual bool set_param(const core::Parameter &param);
protected:
	V4l2Source(log::Log &log_,core::pwThreadBase parent, core::Parameters &parameters)
			IO_THREAD_CONSTRUCTOR;
	bool init_mmap();
	bool init_user();
	bool init_read();
	bool start_capture();
	bool stop_capture();
	bool read_frame();
	static int xioctl(int fd, unsigned long int request, void *arg);
	bool prepare_frame(yuri::ubyte_t *data, yuri::size_t size);

	bool open_file();
	bool query_capabilities();
	bool enum_inputs();
	bool set_input();
	bool set_cropping();
	bool enum_formats();
	bool set_format();
	bool enum_frame_intervals();
	bool set_frame_params();
	bool initialize_capture();
	bool enable_iluminator();

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
	core::pBasicFrame output_frame;
	bool combine_frames;
	std::vector<yuri::uint_t> supported_formats;
	yuri::size_t number_of_inputs;
//	static std::map<yuri::format_t,yuri::uint_t> formats_map;
//	static std::map<std::string, yuri::uint_t> special_formats;
	yuri::size_t fps;
	yuri::size_t frame_duration;
};

}

}
#endif /* V4L2SOURCE_H_ */
