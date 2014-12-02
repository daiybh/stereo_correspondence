/*!
 * @file 		WebStaticResource.h
 * @author 		<Your name>
 * @date 		02.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef WEBSTATICRESOURCE_H_
#define WEBSTATICRESOURCE_H_

#include "yuri/core/thread/IOThread.h"
#include "WebResource.h"

namespace yuri {
namespace webserver {

class WebStaticResource: public core::IOThread, public WebResource
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	WebStaticResource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~WebStaticResource() noexcept;
private:
	
	virtual void run();
	virtual bool set_param(const core::Parameter& param);
	virtual webserver::response_t do_process_request(const webserver::request_t& request) override;
	std::string server_name_;
	std::string path_;
	std::string mime_type_;
	std::string filename_;
	std::string data_string_;
};

} /* namespace web_static */
} /* namespace yuri */
#endif /* WEBSTATICRESOURCE_H_ */
