/*!
 * @file 		WebServer.cpp
 * @author 		<Your name>
 * @date		01.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "WebServer.h"
#include "WebResource.h"
#include "yuri/core/Module.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/version.h"
#include <boost/regex.hpp>

namespace yuri {
namespace webserver {


IOTHREAD_GENERATOR(WebServer)

core::Parameters WebServer::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("WebServer");
	p["socket"]["Socket implementation"]="yuri_tcp";
	p["address"]["Server address"]="0.0.0.0";
	p["server_name"]["Server name"]="webserver";
	p["port"]["Server port"]=8080;
	return p;
}

namespace {
	const std::map<http_code, std::string> common_codes = {
		{http_code::continue_ , "Continue"},
		{http_code::ok, "OK"},
		{http_code::created, "Created"},
		{http_code::accepted, "Accepted"},
		{http_code::no_content, "No content"},
		{http_code::partial, "Partial"},
		{http_code::moved, "Moved Permanently"},
		{http_code::found, "Found"},
		{http_code::see_other, "See Other"},
		{http_code::not_modified, "Not Modified"},
		{http_code::bad_request, "Bad Request"},
		{http_code::unauthorized, "Unauthorized"},
		{http_code::forbidden, "Forbidden"},
		{http_code::not_found, "Not Found"},
		{http_code::gone, "Gone"},
		{http_code::server_error, "Internal Server Error"},
		{http_code::service_unavailable, "Service Unavailable"}
	};

	const std::string default_header =
R"XXX(	<head>
		<meta name="generator" constent="yuri"/>
	</head>
)XXX";
	const std::string default_footer = std::string{"powered by yuri-"}+yuri_version;


	const std::map<http_code, std::string> default_contents = {
			{http_code::not_found, std::string{"<html>\n"}+default_header+"<body>\n<h1>Not found</h1>\n"+default_footer+"\n</html>"},
			{http_code::server_error, std::string{"<html>\n"}+default_header+"<body>\n<h1>Not found</h1>\n"+default_footer+"\n</html>"},
	};

	std::string prepare_response_header(http_code code)
	{
		std::string header = "HTTP/1.1 " + std::to_string(static_cast<int>(code)) + " ";
		auto it = common_codes.find(code);
		if (it == common_codes.end()) return header + "UNKNOWN";
		return header + it->second;
	}

	std::string get_default_contents(http_code code)
	{
		auto it = default_contents.find(code);
		if (it == default_contents.end()) {
			return R"XXX()XXX";
		}
		return it->second;
	}
	const std::string crlf = "\r\n";

	std::map<std::string, pwWebServer> active_servers;
	std::mutex active_servers_mutex;
	void register_server(const std::string& name, pwWebServer server)
	{
		std::unique_lock<std::mutex> _(active_servers_mutex);
		active_servers[name]=server;
	}
}


pwWebServer find_webserver(const std::string& name)
{
	std::unique_lock<std::mutex> _(active_servers_mutex);
	auto it = active_servers.find(name);
	if (it == active_servers.end()) return {};
	return it->second;
}


WebServer::WebServer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("webserver")),server_name_("webserver"),
socket_impl_("yuri_tcp"),address_("0.0.0.0"),port_(8080)
{
	IOTHREAD_INIT(parameters)
	socket_ = core::StreamSocketGenerator::get_instance().generate(socket_impl_, log);
	log[log::info] << "Created socket";
	if (!socket_->bind(address_.c_str(), port_)) {
		log[log::fatal] << "Failed to bind to "+address_+":"+std::to_string(port_);
		throw exception::InitializationFailed("Failed to bind to port "+address_+":"+std::to_string(port_));
	}
	if (!socket_->listen()) {
		log[log::fatal] << "Failed to start listening";
		throw exception::InitializationFailed("Failed to start listening");
	}
}

WebServer::~WebServer() noexcept
{
}

void WebServer::run()
{
	register_server(server_name_, std::dynamic_pointer_cast<WebServer>(get_this_ptr()));
	log[log::info] << "Starting worker thread";
	auto req_thread = std::thread{[&](){response_thread();}};

	while (still_running()) {
		if (socket_->wait_for_data(get_latency())) {
			auto client = socket_->accept();
			log[log::info] << "Connection accepted";
			push_request(std::async(std::launch::async, [=]()mutable{return read_request(client);}));
		}
	}
	log[log::info] << "Joining worker thread";
	req_thread.join();
}


response_t WebServer::find_response(request_t request)
{
	for (const auto& route: routing_) {
		boost::regex url(route.routing_spec);
		if (boost::regex_match(request.url.cbegin(), request.url.cend(), url)) {
			try {
				return route.resource->process_request(request);
			}
			catch (std::runtime_error& e) {
				log[log::info] << "Returning 500 for URL " << request.url << " ("<<e.what()<<")";
				return {http_code::server_error,{},get_default_contents(http_code::not_found)};
			}
		}
	}
	log[log::info] << "Returning 404 for URL " << request.url;
	return {http_code::not_found,{},get_default_contents(http_code::not_found)};
}

void WebServer::response_thread()
{
	log[log::info] << "Helper thread started";
	while(still_running()) {
		auto fr = pop_request();
		if (fr.valid()) {
			try {
				auto status = fr.wait_for(std::chrono::microseconds(get_latency()));
				if (status != std::future_status::ready) {
					push_request(std::move(fr));
				} else {
					auto request = fr.get();
					log[log::info] << "Requested URL: " << request.url;
					response_t response = find_response(request);
					reply_to_client(request.client, std::move(response));
				}
			} catch (std::runtime_error& e) {
				log[log::warning] << "Failed to process connection (" << e.what()<<")";
			}
		}
	}
	log[log::info] << "Helper thread ending";
}

void WebServer::push_request(f_request_t request)
{
	std::unique_lock<std::mutex> _(request_mutex_);
	requests_.push_back(std::move(request));
	// This could be sub-optimal, but it makes the logic easier...
	request_notify_.notify_all();
}
f_request_t WebServer::pop_request()
{
	std::unique_lock<std::mutex> lock(request_mutex_);
	// Don't wait for variable, if there's data ready...
	if (!requests_.empty()) {
		auto r = std::move(requests_.front());
		requests_.pop_front();
		return std::move(r);
	}
	// No data available, so wait for a notification or timeout
	request_notify_.wait_for(lock, std::chrono::microseconds(get_latency()));
	if (!requests_.empty()) {
		auto r = std::move(requests_.front());
		requests_.pop_front();
		return std::move(r);
	}
	// No data even after timeout, nothing to return
	return {};
}


namespace {
bool data_finished(const std::string& data)
{
	const auto len = data.size();
	if (len < 4) return false;
	if (data[len-4] == '\r' && data[len-3] == '\n' &&
		data[len-2] == '\r' && data[len-1] == '\n') return true;
	// Let's support event \n\n
	if (data[len-2] == '\n' && data[len-1] == '\n') return true;
	return false;
}


}
request_t WebServer::read_request(core::socket::pStreamSocket client)
{
	request_t request {{},{},client};
	std::vector<char> data(0);
	data.resize(1024);
	std::string request_string;

	while(!data_finished(request_string) && still_running()) {
		if (client->wait_for_data(get_latency())) {
			auto read = client->receive_data(data);
			if (!read) throw std::runtime_error("Failed to read data");
			request_string.append(data.begin(), data.begin()+read);
		}
	}

	boost::regex url_line("^GET (.*) HTTP/1.1\r?\n");

	boost::smatch what;
	auto start = request_string.cbegin();
	const auto end = request_string.cend();
	if (regex_search(start, end, what, url_line, boost::match_default)) {
		request.url=std::string(what[1].first, what[1].second);
		start = what[0].second;
		boost::regex param_line("([^:]+):([^\r\n]*)\r?\n");
		boost::sregex_iterator i(start, end, param_line, boost::match_default);
		boost::sregex_iterator j;
		while (i != j) {
			const auto& res = *i;
			const auto param_name  =  std::string(res[1].first,res[1].second);
			request.parameters[param_name] = std::string(res[2].first,res[2].second);
			++i;
		}
	} else {
		throw std::runtime_error("Failed to parse url");
	}

	return request;
}

namespace {
inline void fill_header_if_needed(response_t& response, const std::string& name, const std::string& value)
{
	auto it = response.parameters.find(name);
	if (it == response.parameters.end()) {
		response.parameters[name]=value;
	}
}

}

bool WebServer::reply_to_client(core::socket::pStreamSocket& client, response_t response)
{
	fill_header_if_needed(response,"Content-Length",std::to_string(response.data.size()));
	fill_header_if_needed(response,"Server",std::string("yuri-")+yuri_version);

	client->send_data(prepare_response_header(response.code));
	client->send_data(crlf);
	for (const auto&param: response.parameters) {
		client->send_data(param.first);
		client->send_data(std::string(": "));
		client->send_data(param.second);
		client->send_data(crlf);
	}
	client->send_data(crlf);
	client->send_data(response.data);
	return true;
}


bool WebServer::register_resource (const std::string& routing_spec, pWebResource resource)
{
	routing_.push_back({routing_spec, std::move(resource)});
	return true;
}
bool WebServer::set_param(const core::Parameter& param)
{
	if (param.get_name() == "socket") {
		socket_impl_ = param.get<std::string>();
	} else if (param.get_name() == "address") {
		address_ = param.get<std::string>();
	} else if (param.get_name() == "port") {
		port_ = param.get<uint16_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace webserver */
} /* namespace yuri */
