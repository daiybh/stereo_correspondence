/*
 * v4l2_controls.cpp
 *
 *  Created on: 24. 1. 2015
 *      Author: neneko
 */
#include <errno.h>
#include "v4l2_controls.h"
#include "yuri/core/utils.h"
#include <linux/v4l2-controls.h>
#include <sys/ioctl.h>
#include <cstring>

// These controls are in kernel since 2010, but we define theme here nevertheless
#ifndef V4L2_CID_ILLUMINATORS_1
#define V4L2_CID_ILLUMINATORS_1			(V4L2_CID_BASE+37)
#endif

#ifndef V4L2_CID_ILLUMINATORS_2
#define V4L2_CID_ILLUMINATORS_2			(V4L2_CID_BASE+38)
#endif

namespace yuri {

namespace v4l2 {
namespace controls {

namespace {
	int xioctl(int fd, unsigned long int request, void *arg)
	{
		int r;
		while((r = ioctl (fd, request, arg)) < 0) {
			if (r==-1 && errno==EINTR) continue;
			break;
		}
		return r;
	}

	enum class control_support_t {
			supported,
			not_supported,
			disabled,
	};

	struct control_state_t {
		control_support_t supported;
		int32_t value;
		int32_t min_value;
		int32_t max_value;
		std::string name;
	};

	template<class T>
	int32_t get_event_value(const control_state_t& state, const T& value)
	{
		if (value.range_specified()) {
			const auto tgt_range = state.max_value - state.min_value;
			const auto src_range = value.get_max_value() - value.get_min_value();
			return static_cast<int32_t>(static_cast<double>(value.get_value() - value.get_min_value()) * tgt_range / src_range) + state.min_value;
		}
		return value.get_value();
	}

	const std::map<int, std::string> user_controls = {
	{V4L2_CID_BRIGHTNESS, "brightness"},
	{V4L2_CID_CONTRAST, "contrast"},
	{V4L2_CID_SATURATION, "saturation"},
	{V4L2_CID_HUE, "hue"},
	{V4L2_CID_AUDIO_VOLUME, "audio_volume"},
	{V4L2_CID_AUDIO_BALANCE, "audio_balance"},
	{V4L2_CID_AUDIO_BASS, "audio_bass"},
	{V4L2_CID_AUDIO_TREBLE, "audio_treble"},
	{V4L2_CID_AUDIO_MUTE, "audio_mute"},
	{V4L2_CID_AUDIO_LOUDNESS, "audio_loudness"},
	{V4L2_CID_AUTO_WHITE_BALANCE, "auto_white_balance"},
	{V4L2_CID_DO_WHITE_BALANCE, "do_white_balance"},
	{V4L2_CID_RED_BALANCE, "red_balance"},
	{V4L2_CID_BLUE_BALANCE, "blue_balance"},
	{V4L2_CID_GAMMA, "gamma"},
	{V4L2_CID_EXPOSURE, "exposure"},
	{V4L2_CID_AUTOGAIN, "autogain"},
	{V4L2_CID_GAIN, "gain"},
	{V4L2_CID_HFLIP, "hflip"},
	{V4L2_CID_VFLIP, "vflip"},
	{V4L2_CID_POWER_LINE_FREQUENCY, "power_line_frequency"},
	{V4L2_CID_HUE_AUTO, "hue_auto"},
	{V4L2_CID_WHITE_BALANCE_TEMPERATURE, "white_ballance_temperature"},
	{V4L2_CID_SHARPNESS, "sharpness"},
	{V4L2_CID_BACKLIGHT_COMPENSATION, "backlight_compensation"},
	{V4L2_CID_CHROMA_AGC, "chroma_agc"},
	{V4L2_CID_COLOR_KILLER, "color_killer"},
	{V4L2_CID_COLORFX, "color_fx"},
	{V4L2_CID_AUTOBRIGHTNESS, "auto_brightness"},
	{V4L2_CID_BAND_STOP_FILTER, "band_stop_filter"},
	{V4L2_CID_ROTATE, "rotate"},
	{V4L2_CID_BG_COLOR, "bg_color"},
	{V4L2_CID_CHROMA_GAIN, "chroma_gain"},
	{V4L2_CID_ILLUMINATORS_1, "illuminator"},
	{V4L2_CID_ILLUMINATORS_2, "illuminator2"}
	};

	std::string get_user_control_name(int control)
	{
		auto it = user_controls.find(control);
		if (it == user_controls.end()) return {};
		return it->second;
	}
	int get_user_control_by_name(std::string name)
	{
		for (const auto& ctrl: user_controls) {
			if (iequals(ctrl.second, name)) return ctrl.first;
		}
		return 0;
	}

	control_state_t is_control_supported_impl(int fd, uint32_t id, log::Log& log)
	{
		struct v4l2_queryctrl queryctrl;
		std::memset (&queryctrl, 0, sizeof (queryctrl));
		queryctrl.id=id;

		if (xioctl (fd, VIDIOC_QUERYCTRL, &queryctrl) < 0) {
			log[log::debug]<< "Control " << queryctrl.name << " not supported";
			return {control_support_t::not_supported, 0, 0, 0, {}};
		} else if (queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
			log[log::debug]<< "Control " << queryctrl.name << " disabled";
			return {control_support_t::disabled, 0, 0, 0, {}};
		}
		return {control_support_t::supported, 0, queryctrl.minimum, queryctrl.maximum, reinterpret_cast<char*>(queryctrl.name)};
	}

	control_state_t is_user_control_supported(int fd, uint32_t id, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported == control_support_t::supported) {
			struct v4l2_control control;
			std::memset (&control, 0, sizeof (control));
			control.id = id;
			if (xioctl (fd, VIDIOC_G_CTRL, &control)>=0) {
				state.value = control.value;
			}
		}
		return state;
	}
	bool set_user_control_impl(int fd, uint32_t id, control_state_t state, int32_t value, log::Log& log)
	{
		struct v4l2_control control;
		std::memset (&control, 0, sizeof (control));
		control.id = id;
		control.value = clip_value(value, state.min_value, state.max_value);

		if (xioctl (fd, VIDIOC_S_CTRL, &control) < 0) {
			log[log::debug]<< "Failed to enable control " << state.name;
			return false;

		} else {
			control.value = 0;
			if (xioctl (fd, VIDIOC_G_CTRL, &control)>=0) {
				log[log::info] << "Control " << state.name << " set to " << control.value;
			} else {
				log[log::warning] << "Failed to set value for " << state.name;
				return false;
			}
		}
		return true;
	}

	bool set_user_control(int fd, uint32_t id, bool value, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}
		return set_user_control_impl(fd, id, std::move(state), value?state.max_value:state.min_value, log);
	}
	bool set_user_control(int fd, uint32_t id, int32_t value, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}
		return set_user_control_impl(fd, id, std::move(state), value, log);
	}


	bool set_user_control(int fd, uint32_t id, const event::pBasicEvent& event, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}

		if (event->get_type() == event::event_type_t::boolean_event) {
			return set_user_control_impl(fd, id, std::move(state), event::get_value<event::EventBool>(event), log);
		}
		if (event->get_type() == event::event_type_t::integer_event) {
			auto value = get_event_value(state, *dynamic_pointer_cast<event::EventInt>(event));
			return set_user_control_impl(fd, id, std::move(state), value, log);
		}
		if (event->get_type() == event::event_type_t::double_event) {
			auto value = get_event_value(state, *dynamic_pointer_cast<event::EventDouble>(event));
			return set_user_control_impl(fd, id, std::move(state), value, log);
		}
		return false;
		//return set_control_impl(fd, id, std::move(state), value, log);
	}

	const std::map<int, std::string> camera_controls = {
	{V4L2_CID_EXPOSURE_AUTO, "exposure_auto"},
	{V4L2_CID_EXPOSURE_ABSOLUTE, "exposure_absolute"},
	{V4L2_CID_EXPOSURE_AUTO_PRIORITY, "exposure_auto_priority"},
	{V4L2_CID_PAN_RELATIVE, "pan_relative"},
	{V4L2_CID_TILT_RELATIVE, "tilt_relative"},
	{V4L2_CID_PAN_RESET, "pan_reset"},
	{V4L2_CID_TILT_RESET, "tilt_reset"},
	{V4L2_CID_PAN_ABSOLUTE, "pan_absolute"},
	{V4L2_CID_TILT_ABSOLUTE, "tilt_absolute"},
	{V4L2_CID_FOCUS_ABSOLUTE, "focus_absolute"},
	{V4L2_CID_FOCUS_RELATIVE, "focus_relative"},
	{V4L2_CID_FOCUS_AUTO, "focus_auto"},
	{V4L2_CID_ZOOM_ABSOLUTE, "zoom_absolute"},
	{V4L2_CID_ZOOM_RELATIVE, "zoom_relative"},
	{V4L2_CID_ZOOM_CONTINUOUS, "zoom_continuous"},
	{V4L2_CID_PRIVACY, "privacy"},
	{V4L2_CID_IRIS_ABSOLUTE, "iris_absolute"},
	{V4L2_CID_IRIS_RELATIVE, "iris_relative"},
	{V4L2_CID_AUTO_EXPOSURE_BIAS, "exposure_bias_auto"},
	{V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE, "white_balance_preset_auto"},
	{V4L2_CID_WIDE_DYNAMIC_RANGE, "wide_dynamic_range"},
	{V4L2_CID_IMAGE_STABILIZATION, "image_stabilization"},
	{V4L2_CID_ISO_SENSITIVITY, "iso_sensitivity"},
	{V4L2_CID_ISO_SENSITIVITY_AUTO, "iso_sensitivity_auto"},
	{V4L2_CID_EXPOSURE_METERING, "exposure_metering"},
	{V4L2_CID_SCENE_MODE, "scene_mode"},
	{V4L2_CID_3A_LOCK, "3a_lock"}};


	std::string get_camera_control_name(int control)
	{
		auto it = camera_controls.find(control);
		if (it == camera_controls.end()) return {};
		return it->second;
	}
	int get_camera_control_by_name(std::string name)
	{
		for (const auto& ctrl: camera_controls) {
			if (iequals(ctrl.second, name)) return ctrl.first;
		}
		return 0;
	}


	control_state_t is_camera_control_supported(int fd, uint32_t id, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported == control_support_t::supported) {
			v4l2_ext_control control {id, 0, {0}, {0}};
			v4l2_ext_controls controls {V4L2_CID_CAMERA_CLASS, 1, 0, {0,0}, &control};
			if (xioctl (fd, VIDIOC_G_EXT_CTRLS, &controls)>=0) {
				state.value = control.value;
			}
		}
		return state;
	}


	bool set_camera_control_impl(int fd, uint32_t id, control_state_t state, int32_t value, log::Log& log)
	{
		v4l2_ext_control control {id, 0, {0}, {clip_value(value, state.min_value, state.max_value)}};
		v4l2_ext_controls controls {V4L2_CID_CAMERA_CLASS, 1, 0, {0,0}, &control};

		if (xioctl (fd, VIDIOC_S_EXT_CTRLS, &controls) < 0) {
			log[log::debug]<< "Failed to enable control " << state.name;
			return false;

		} else {
			control.value = 0;
//			if (xioctl (fd, VIDIOC_G_EXT_CTRLS, &control)>=0) {
				log[log::info] << "Control " << state.name << " set to " << control.value;
//			} else {
//				log[log::warning] << "Failed to set value for " << state.name;
//				return false;
//			}
		}
		return true;
	}

	bool set_camera_control(int fd, uint32_t id, bool value, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}
		return set_camera_control_impl(fd, id, std::move(state), value?state.max_value:state.min_value, log);
	}
	bool set_camera_control(int fd, uint32_t id, int32_t value, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}
		return set_camera_control_impl(fd, id, std::move(state), value, log);
	}


	bool set_camera_control(int fd, uint32_t id, const event::pBasicEvent& event, log::Log& log)
	{
		auto state = is_control_supported_impl(fd, id, log);
		if (state.supported != control_support_t::supported) {
			return false;
		}

		if (event->get_type() == event::event_type_t::boolean_event) {
			return set_camera_control_impl(fd, id, std::move(state), event::get_value<event::EventBool>(event), log);
		}
		if (event->get_type() == event::event_type_t::integer_event) {
			auto value = get_event_value(state, *dynamic_pointer_cast<event::EventInt>(event));
			return set_camera_control_impl(fd, id, std::move(state), value, log);
		}
		if (event->get_type() == event::event_type_t::double_event) {
			auto value = get_event_value(state, *dynamic_pointer_cast<event::EventDouble>(event));
			return set_camera_control_impl(fd, id, std::move(state), value, log);
		}
		return false;
		//return set_control_impl(fd, id, std::move(state), value, log);
	}


	template<class T>
	bool set_control_generic(int fd, int id, const T& value, log::Log& log)
	{
		if (contains(user_controls, id)) return set_user_control(fd, id, value, log);
		if (contains(camera_controls, id)) return set_camera_control(fd, id, value, log);
		return false;
	}

	template<class T>
	bool set_control_generic(int fd, const std::string& name, const T& value, log::Log& log)
	{
		auto id = get_user_control_by_name(name);
		if (id) return set_user_control(fd, id, value, log);
		auto cid = get_camera_control_by_name(name);
		if (cid) return set_camera_control(fd, id, value, log);
		return false;
	}
}


std::vector<control_info> get_control_list(int fd, log::Log& log)
{
	std::vector<control_info> ctrls;
	for (const auto& c: user_controls) {
		auto state = is_user_control_supported(fd, c.first, log);
		if (state.supported == control_support_t::supported) {
			ctrls.push_back({c.first, state.name, c.second, state.value, state.min_value, state.max_value});
		}
	}
	for (const auto& c: camera_controls) {
		auto state = is_camera_control_supported(fd, c.first, log);
		if (state.supported == control_support_t::supported) {
			ctrls.push_back({c.first, state.name, c.second, state.value, state.min_value, state.max_value});
		}
	}
	return ctrls;
}


std::string get_control_name(int control)
{
	auto n = get_user_control_name(control);
	if (!n.empty()) return n;
	return get_camera_control_name(control);
}
int get_control_by_name(std::string name)
{
	if (auto id = get_user_control_by_name(name)) {
		return id;
	}
	return get_camera_control_by_name(name);
}


bool set_control(int fd, int id, bool value, log::Log& log)
{
	return set_control_generic(fd, id, value, log);
}
bool set_control(int fd, int id, int32_t value, log::Log& log)
{
	return set_control_generic(fd, id, value, log);
}
bool set_control(int fd, int id, const event::pBasicEvent& value, log::Log& log)
{
	return set_control_generic(fd, id, value, log);
}

bool set_control(int fd, const std::string& name, bool value, log::Log& log)
{
	return set_control_generic(fd, name, value, log);
}
bool set_control(int fd, const std::string& name, int32_t value, log::Log& log)
{
	return set_control_generic(fd, name, value, log);
}
bool set_control(int fd, const std::string& name, const event::pBasicEvent& value, log::Log& log)
{
	return set_control_generic(fd, name, value, log);
}





}
}
}
