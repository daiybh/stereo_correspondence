/*!
 * @file 		EventDevice.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		11.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "EventDevice.h"
#include "yuri/core/Module.h"
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
namespace yuri {
namespace event_device {

IOTHREAD_GENERATOR(EventDevice)


core::Parameters EventDevice::configure()
{
	auto p = core::IOThread::configure();
	p.set_description("EventDevice");
	p["device"]["Path to the device"]="/dev/input/event0";
	p["min_fuzz"]["Minimal value for fuzz. Increase if the device generates noised valued"]=0;
	return p;
}


namespace {
	inline bool test_bit(int idx, const uint8_t* array) {
		return (array[idx/8] & (1<<(idx%8)));

	}
	std::string get_axis_name(int code) {
		static std::map<int, std::string> names {
			{ABS_X, "x"},
			{ABS_Y, "y"},
			{ABS_Z, "z"},
			{ABS_RX, "rx"},
			{ABS_RY, "ry"},
			{ABS_RZ, "rz"},
			{ABS_THROTTLE, "throttle"},
			{ABS_RUDDER, "rudder"},
			{ABS_WHEEL, "wheel"},
			{ABS_GAS, "gas"},
			{ABS_BRAKE, "brake"},
			{ABS_HAT0X, "hat0x"},
			{ABS_HAT0Y, "hat0y"},
			{ABS_HAT1X, "hat1x"},
			{ABS_HAT1Y, "hat1y"},
			{ABS_HAT2X, "hat2x"},
			{ABS_HAT2Y, "hat2y"},
			{ABS_HAT3X, "hat3x"},
			{ABS_HAT3Y, "hat3y"},
			{ABS_PRESSURE, "pressure"},
			{ABS_DISTANCE, "distance"},
			{ABS_TILT_X, "tilt_x"},
			{ABS_TILT_Y, "tile_y"},
			{ABS_TOOL_WIDTH, "tool_width"},
			{ABS_VOLUME, "volume"},
			{ABS_MISC, "misc"}
		};
		auto n = names.find(code);
		if (n == names.end()) return "unknown";
		return n->second;
	}
}

EventDevice::EventDevice(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("event_device")),
event::BasicEventProducer(log),
handle_(-1),min_fuzz_(0)
{
	IOTHREAD_INIT(parameters)
	if ((handle_ = ::open(device_path_.c_str(), O_RDONLY)) < 0 ) {
		throw exception::InitializationFailed("Failed to open device "+device_path_);
	}
	uint8_t abs_bitmask[ABS_MAX/8 + 1];
	::ioctl(handle_, EVIOCGBIT(EV_ABS, sizeof(abs_bitmask)), abs_bitmask);

	for (int i=0;i<ABS_MAX;++i) {
		if (test_bit(i,abs_bitmask)) {
			auto& info = axis_info_[i];
			info.code = i;
			::ioctl(handle_, EVIOCGABS(i), &(info.info));
			info.name = get_axis_name(i);
			info.last_value=info.info.value;
			info.info.fuzz=std::max(info.info.fuzz,min_fuzz_);
			log[log::info] << "Found axis " << info.name;
		}
	}
}

EventDevice::~EventDevice() noexcept
{
}

void EventDevice::run()
{
	using namespace yuri::event;
//	IO_THREAD_PRE_RUN
	while (still_running()) {
		pollfd fd = {handle_, POLLIN, 0};
		::poll(&fd, 1, get_latency().value/1000);
		if (fd.revents & POLLIN) {
			input_event ev;
			if (::read(handle_,&ev,sizeof(input_event))>0) {
				if (ev.type == EV_ABS) {
					pBasicEvent event;
					auto iter = axis_info_.find(ev.code);
					if (iter==axis_info_.end()) continue;
					auto& info = iter->second;

					if (std::abs(info.last_value-ev.value)<=info.info.fuzz) continue;

					if (info.info.maximum != info.info.minimum) {
						event = std::make_shared<EventInt>(ev.value, info.info.minimum, info.info.maximum);
					} else {
						event = std::make_shared<EventInt>(ev.value);
					}
					log[log::debug] << "Emiting value " << ev.value << " for " << info.name << " (last value: " << info.last_value<<")";
					info.last_value = ev.value;
					emit_event(info.name, event);
				}
			}
		}

	}
//	IO_THREAD_POST_RUN
}
bool EventDevice::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(device_path_, "device")
			(min_fuzz_, "min_fuzz"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace event_device */
} /* namespace yuri */
