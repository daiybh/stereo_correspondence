/*!
 * @file 		v4l2_device.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef V4L2_DEVICE_H_
#define V4L2_DEVICE_H_

#include <vector>
#include <cstdint>
#include <string>
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/time_types.h"
#include "yuri/core/utils/uvector.h"
#include "yuri/log/Log.h"
#include "v4l2_common.h"

namespace yuri {
namespace v4l2 {

std::vector<std::string> enum_v4l2_devices();

enum v4l2_caps: uint8_t {
	streaming	= 0x01,
	read_write	= 0x02,
};
struct v4l2_device_info {
	std::string name;
	std::string driver;
	std::string driver_version;
	std::string bus;
	uint8_t 	caps;
};

enum v4l2_input_states :long {
	no_power 	= 0x00001,
	no_signal 	= 0x00002,
	no_color 	= 0x00004,
	horiz_flip	= 0x00008,
	vert_flip	= 0x00010,
	not_camera	= 0x00020
};

struct v4l2_input_info {
	std::string name;
	long state;
};

struct v4l2_std_info {
	std::string name;
	uint64_t id;
};

struct v4l2_format_info {
	size_t imagesize;
	resolution_t resolution;
};
struct v4l2_device {
	v4l2_device(const std::string& path);
	v4l2_device(const v4l2_device&) = delete;
	v4l2_device& operator=(const v4l2_device&) = delete;
	v4l2_device(v4l2_device&& rhs) noexcept;
	v4l2_device& operator=(v4l2_device&& rhs) noexcept;
	~v4l2_device() noexcept;

	int get() const;
	v4l2_device_info get_info();

	std::vector<v4l2_input_info> enum_inputs();
	bool set_input(int index);

	std::vector<v4l2_std_info> enum_standards();
//	bool set_standard()

	std::vector<uint32_t> enum_formats();
	v4l2_format_info set_format(uint32_t format, resolution_t res);

	std::vector<resolution_t> enum_resolutions(uint32_t fmt);

	std::vector<fraction_t> enum_fps(uint32_t fmt, resolution_t res);
	fraction_t set_fps(fraction_t fps);

	bool set_default_cropping();


	std::vector<buffer_t> init_mmap();
	std::vector<buffer_t> init_user(size_t imagesize);
	std::vector<buffer_t> init_read(size_t imagesize);

	bool initialize_capture(size_t imagesize, capture_method_t method, log::Log& log);


	bool start_capture();
	bool stop_capture();
	bool read_frame(std::function<bool(uint8_t*, size_t)>);

	bool wait_for_data(duration_t duration);

	bool is_running() const { return running_; }

	control_state_t is_control_supported(uint32_t id);
	control_state_t is_user_control_supported(uint32_t id);
	control_state_t is_camera_control_supported(uint32_t id);

	bool set_user_control(uint32_t id, control_state_t state, int32_t value);
	bool set_camera_control(uint32_t id, control_state_t state, int32_t value);
private:
	int fd_;
	v4l2_device_info info_;
	capture_method_t method_;
	size_t imagesize_;
	std::vector<buffer_t> buffers_;
	bool running_;
};

}
}



#endif /* V4L2_DEVICE_H_ */
