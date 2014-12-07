/*!
 * @file 		WebResource.cpp
 * @author 		<Your name>
 * @date		02.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "WebResource.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace webserver {


WebResource::WebResource(const log::Log &log_):log_res(log_),registered_(false)
{
}

WebResource::~WebResource() noexcept
{
}

bool WebResource::register_to_server(const std::string& server_name, const std::string& routing_spec, pWebResource resource)
{
	if (registered_) return true;
	auto server_ptr = webserver::find_webserver(server_name);
	if (!server_ptr.expired()) {
		auto server = server_ptr.lock();
		return (registered_ = server->register_resource(routing_spec, resource));
	}
	return false;
}

webserver::response_t WebResource::process_request(const webserver::request_t& request)
{
	return do_process_request(request);
}

} /* namespace webresource */
} /* namespace yuri */
