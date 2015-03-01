/*!
 * @file 		WebControlResource.cpp
 * @author 		Zdenek Travnicek <travnicek@cesnet.cz>
 * @date		02.12.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD or GPL License,
 * 				see /doc/LICENSE.txt for details
 *
 */

#include "WebControlResource.h"
#include "yuri/core/Module.h"
#include "web_exceptions.h"
#include "yuri/event/BasicEventParser.h"
namespace yuri {
namespace webserver {


IOTHREAD_GENERATOR(WebControlResource)


core::Parameters WebControlResource::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("WebControlResource");
	p["server_name"]["Name of server"]="webserver";
	p["path"]["Path prefix"]="/control";
	p["redirect"]["Redirect target"]="/";
	return p;
}


WebControlResource::WebControlResource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("web_control")),WebResource(log),BasicEventProducer(log),
server_name_("webserver"),path_("/control"),redirect_path_("/")
{
	IOTHREAD_INIT(parameters)
}

WebControlResource::~WebControlResource() noexcept
{
}

void WebControlResource::run()
{
	while (still_running() && !register_to_server(server_name_, path_, std::dynamic_pointer_cast<WebResource>(get_this_ptr()))) {
		sleep(10_ms);
	}
	log[log::info] << "Registered to server";
	while (still_running()) {
		sleep(100_ms);
	}
}

webserver::response_t WebControlResource::do_process_request(const webserver::request_t& request)
{
	log[log::info] << "Responding";
//	auto prefix = request.url.path.substr(0, path_.size());
//	if (prefix != path_) throw std::runtime_error("Wrong path");
//	if ()
	for (const auto&x : request.url.params) {
//		log[log::info] << "Param: " << x.first << " -> " << x.second;
		const auto& cmd = x.first;
		const auto& val = x.second;
		if (!val.empty()) {
			log[log::info] << "Sending command " << cmd << " = " << val;
			auto event = event::BasicEventParser::parse_expr(log,val,std::map<std::string,event::pBasicEvent>{});
			if (!event) {
				event = std::make_shared<event::EventString>(val);
			}
			emit_event(cmd, event);
		} else {
			log[log::info] << "Sending command " << cmd;
			emit_event(cmd);
		}
	}
	throw redirect_to(redirect_path_);
}
bool WebControlResource::set_param(const core::Parameter& param)
{
	if (param.get_name() == "server_name") {
		server_name_ = param.get<std::string>();
	} else if (param.get_name() == "path") {
		path_ = param.get<std::string>();
	} else if (param.get_name() == "redirect") {
		redirect_path_ = param.get<std::string>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace web_static */
} /* namespace yuri */
