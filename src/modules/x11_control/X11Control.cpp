/*!
 * @file 		X11Control.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.11.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "X11Control.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils/assign_events.h"
namespace yuri {
namespace x11_control {


IOTHREAD_GENERATOR(X11Control)

MODULE_REGISTRATION_BEGIN("x11_control")
		REGISTER_IOTHREAD("x11_control",X11Control)
MODULE_REGISTRATION_END()

core::Parameters X11Control::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("X11Control");
	p["display"]["X11 display string"]="";
	return p;
}


X11Control::X11Control(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("x11_control")),
BasicEventConsumer(log),
dpy_(nullptr, [](xcb_connection_t*p){xcb_disconnect(p);})
{
	IOTHREAD_INIT(parameters)
	int screen = 0;
	dpy_.reset(xcb_connect(display_.empty()?nullptr:display_.c_str(), &screen));
	if (!dpy_.get() || xcb_connection_has_error(dpy_.get()) != 0) {
		throw exception::InitializationFailed("Failed to connect to display \""+display_+"\"");
	}

	log[log::info] << "Connected to display, using screen " << screen;
	auto setup = xcb_get_setup(dpy_.get());
	if (!setup) throw exception::InitializationFailed("Failed to query setup for current connection");
	auto iter = xcb_setup_roots_iterator(setup);
	if (screen > iter.rem) {
		throw exception::InitializationFailed("Failed to get current screen info (only " + std::to_string(iter.rem) + "screens)");
	}
	for (int i = 0; i < screen; ++i) {
		xcb_screen_next(&iter);
	}
//	if (iter.index != screen)
//		throw exception::InitializationFailed("Failed to get current screen info");
	screen_ = iter.data;

}

X11Control::~X11Control() noexcept
{
}

void X11Control::run()
{
	while(still_running())
	{
		wait_for_events(get_latency());
		process_events();
	}
}

bool X11Control::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(display_, "display"))
		return true;
	return core::IOThread::set_param(param);
}

void X11Control::fake_key(bool press, uint8_t key)
{
	auto cookie = xcb_test_fake_input(dpy_.get(),
			press?XCB_KEY_PRESS:XCB_KEY_RELEASE,
			key,
			XCB_CURRENT_TIME,
			screen_->root,
			0,0,0);
	auto err = xcb_request_check(dpy_.get(), cookie);
	if (err) {
		log[log::warning] << "Failed to submit event";
		free(err);
	}
	xcb_flush(dpy_.get());

}

bool X11Control::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	auto key = event::lex_cast_value<uint8_t>(event);

	if (event_name == "key") {
		log[log::info] << "Pressing key: " << key << "(" << static_cast<int>(key) << ")";
		fake_key(true, key);
		fake_key(false, key);
	} else if (event_name == "key_down") {
		log[log::info] << "key down: " << key << "(" << static_cast<int>(key) << ")";
		fake_key(true, key);
	} else if (event_name == "key_up") {
		log[log::info] << "key up: " << key << "(" << static_cast<int>(key) << ")";
		fake_key(false, key);
	} else {
		log[log::info] << "Unknown event";
	}
	return false;
}

} /* namespace x11_control */
} /* namespace yuri */
