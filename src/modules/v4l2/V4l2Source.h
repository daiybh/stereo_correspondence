/*!
 * @file 		V4l2Source.h
 * @author 		Zdenek Travnicek
 * @date 		17.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef V4L2SOURCE_H_
#define V4L2SOURCE_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"
#include "v4l2_controls.h"

#include <iostream>
#include <linux/videodev2.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <malloc.h>

namespace yuri {

namespace v4l2 {

class V4l2Source: public core::IOThread, public event::BasicEventConsumer {
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
	V4l2Source(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~V4l2Source() noexcept;
	virtual void run();
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	virtual bool set_param(const core::Parameter &param);

	bool init_mmap();
	bool init_user();
	bool init_read();
	bool start_capture();
	bool stop_capture();
	bool read_frame();
	bool prepare_frame(uint8_t *data, yuri::size_t size);
//	virtual bool step() override;
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
	bool enum_controls();
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;


	std::string filename;
	int fd;
	v4l2_capability cap;
	v4l2_format fmt;
	dimension_t /*width,height,*/imagesize;
	resolution_t resolution;
	uint pixelformat;
	methods method;
	bool illumination;
	buffer_t *buffers;
	size_t no_buffers;
	size_t input_number;
	size_t buffer_free;
	core::pFrame output_frame;
	bool combine_frames;
	std::vector<int> supported_formats;
	size_t number_of_inputs;

	fraction_t fps;
	duration_t frame_duration;

	std::vector<controls::control_info> controls_;
};

}

}
#endif /* V4L2SOURCE_H_ */
