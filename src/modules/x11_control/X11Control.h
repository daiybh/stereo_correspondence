/*!
 * @file 		X11Control.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.11.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef X11CONTROL_H_
#define X11CONTROL_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include <xcb/xtest.h>

namespace yuri {
namespace x11_control {

class X11Control: public core::IOThread, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	X11Control(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~X11Control() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	void fake_key(bool press, uint8_t key);
private:
	std::string display_;
	std::unique_ptr<xcb_connection_t, std::function<void(xcb_connection_t*)>> dpy_;
	xcb_screen_t *screen_;
};

} /* namespace x11_control */
} /* namespace yuri */
#endif /* X11CONTROL_H_ */
