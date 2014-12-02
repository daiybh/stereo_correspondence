/*!
 * @file 		WebServer.h
 * @author 		<Your name>
 * @date 		01.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef WEBSERVER_H_
#define WEBSERVER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/socket/StreamSocket.h"

namespace yuri {
namespace webserver {

enum class http_code {
	continue_ = 100,
	ok = 200,
	created = 201,
	accepted = 202,
	no_content = 204,
	partial = 206,
	moved = 301,
	found = 302,
	see_other = 303,
	not_modified = 304,
	bad_request = 400,
	unauthorized = 401,
	forbidden = 403,
	not_found = 404,
	gone = 410,
	server_error = 500,
	service_unavailable = 503
};


using parameters_t = std::map<std::string, std::string>;
struct request_t
{
	std::string url;
	parameters_t parameters;
};

struct response_t
{
	http_code code;
	parameters_t parameters;
	std::string data;
};

class WebServer: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	WebServer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~WebServer() noexcept;
private:
	
	virtual void run();
	virtual bool set_param(const core::Parameter& param);

	request_t read_request(core::socket::pStreamSocket& socket);
	bool reply_to_client(core::socket::pStreamSocket& socket, const response_t& response);
	std::string socket_impl_;
	std::string address_;
	uint16_t port_;
	core::socket::pStreamSocket socket_;


};

} /* namespace webserver */
} /* namespace yuri */
#endif /* WEBSERVER_H_ */
