/*!
 * @file 		FlyCap.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FLYCAP_H_
#define FLYCAP_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/thread/InputThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "flycap_c_helpers.h"

#include <array>
namespace yuri {
namespace flycap {

class FlyCap: public core::IOThread, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	static std::vector<core::InputDeviceInfo> enumerate();
	FlyCap(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FlyCap() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
private:
	resolution_t resolution_;
	geometry_t geometry_;
	format_t format_;
	size_t fps_;

	size_t index_;
	size_t serial_;
	bool keep_format_;
	bool embedded_framecounter_;
	int custom_;
	flycap_camera_t ctx_;

	float shutter_time_;
	float gain_;
	float brightness_;
	float gamma_;
	float exposure_;
	float sharpness_;

	bool trigger_;
	unsigned int trigger_mode_;
	unsigned int trigger_source_;
	std::array<uint8_t,4> gpio_directions_;
	std::array<bool,4> strobes_;
	std::array<bool,4> polarities_;
	std::array<float,4> delays_;
	std::array<float,4> durations_;

	bool pause_;
};

} /* namespace flycap */
} /* namespace yuri */
#endif /* FLYCAP_H_ */
