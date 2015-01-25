/*!
 * @file 		v4l2_controls.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.1.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef V4L2_CONTROLS_H_
#define V4L2_CONTROLS_H_

#include <linux/videodev2.h>
#include "yuri/event/EventHelpers.h"
#include "yuri/log/Log.h"
namespace yuri {

namespace v4l2 {
namespace controls {

struct control_info {
	int id;
	std::string name;
	std::string short_name;
	int32_t value;
	int32_t min_value;
	int32_t max_value;
};

std::vector<control_info> get_control_list(int fd, log::Log& log);

std::string get_control_name(int control);
int get_control_by_name(std::string name);

bool set_control(int fd, int id, bool value, log::Log& log);
bool set_control(int fd, int id, int32_t value, log::Log& log);
bool set_control(int fd, int id, const event::pBasicEvent& value, log::Log& log);
bool set_control(int fd, const std::string& name, bool value, log::Log& log);
bool set_control(int fd, const std::string& name, int32_t value, log::Log& log);
bool set_control(int fd, const std::string& name, const event::pBasicEvent& value, log::Log& log);

}
}
}

#endif /* V4L2_CONTROLS_H_ */
