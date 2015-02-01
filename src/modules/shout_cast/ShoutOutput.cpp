/*!
 * @file 		ShoutOutput.cpp
 * @author 		<Your name>
 * @date		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "ShoutOutput.h"
#include "yuri/core/Module.h"
#include "yuri/version.h"
#include "yuri/core/frame/compressed_frame_types.h"
namespace yuri {
namespace shout_cast {


IOTHREAD_GENERATOR(ShoutOutput)

MODULE_REGISTRATION_BEGIN("shout_output")
		REGISTER_IOTHREAD("shout_output",ShoutOutput)
MODULE_REGISTRATION_END()

core::Parameters ShoutOutput::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("ShoutOutput");
	p["server"]["Server to connect to"]="localhost";
	p["port"]["Port"]=8000;
	p["mount"]["Mount point"]="";
	p["user"]["Username"]="source";
	p["password"]["password"]="";
	p["agent"]["User agent"]=std::string("Yuri ")+yuri_version;
	p["protocol"]["Protocol to connect to server (HTTP, ICY, XAUDIOCAST)"]="HTTP";
	p["description"]["Stream description"]="";
	p["title"]["Stream title"]="";
	p["sync"]["Sync with server. Disable it, if you experience problems with playback."]=true;
	p["url"]["URL to announce with the stream."]="";
	return p;
}

namespace {
void throw_shout_call(int ret, shout_t* shout, const std::string& msg) {
	if (ret != SHOUTERR_SUCCESS) {
		throw exception::InitializationFailed(msg+": "+shout_get_error(shout));
	}
}
auto compare_istr = [](const std::string& a, const std::string&b){return iless(a,b);};
std::map<std::string, int, decltype(compare_istr)> shout_protocol_str = {{
		{"http",SHOUT_PROTOCOL_HTTP},
		{"xaudiocast",SHOUT_PROTOCOL_XAUDIOCAST},
		{"icy",SHOUT_PROTOCOL_ICY}
},compare_istr};



}

ShoutOutput::ShoutOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("shout_cast")),port_(8000),
user_("source"),agent_(std::string("Yuri ")+yuri_version),protocol_(SHOUT_PROTOCOL_HTTP),
sync_(true)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::compressed_frame::ogg});
	if (server_.empty()) throw exception::InitializationFailed("No server specified!");
	if (mount_.empty()) throw exception::InitializationFailed("No mountpoint specified!");
	if (user_.empty()) throw exception::InitializationFailed("No username specified!");
	if (password_.empty()) throw exception::InitializationFailed("No password specified!");
	if (agent_.empty()) throw exception::InitializationFailed("No user agent!");

	log[log::info] << "Gonna connect using " << user_ << ":" << password_;
	shout_init();
	log[log::info] << "Using libshout version " << shout_version(nullptr, nullptr, nullptr);
	shout_ ={shout_new(), [](shout_t*p){shout_close(p);}};
	throw_shout_call(shout_set_host(shout_.get(), server_.c_str()),shout_.get(),"Failed to set hostname");

	throw_shout_call(shout_set_protocol(shout_.get(), protocol_),shout_.get(),"Failed to set protocol");
	throw_shout_call(shout_set_port(shout_.get(), port_),shout_.get(),"Failed to set port");
	throw_shout_call(shout_set_mount(shout_.get(), mount_.c_str()),shout_.get(),"Failed to set mountpoint");
	throw_shout_call(shout_set_host(shout_.get(), server_.c_str()),shout_.get(),"Failed to set hostname");

	throw_shout_call(shout_set_password(shout_.get(), password_.c_str()),shout_.get(),"Failed to set password");
	throw_shout_call(shout_set_user(shout_.get(), user_.c_str()),shout_.get(),"Failed to set username");

	throw_shout_call(shout_set_format(shout_.get(), SHOUT_FORMAT_OGG),shout_.get(),"Failed to set hostname");
	throw_shout_call(shout_set_agent(shout_.get(), agent_.c_str()),shout_.get(), "Failed to set user agent");

	if (!url_.empty()) {
		throw_shout_call(shout_set_url(shout_.get(), url_.c_str()),shout_.get(), "Failed to set url");
	}

	shout_metadata_t *m = shout_metadata_new();
	if (!description_.empty()) {
		throw_shout_call(shout_set_description(shout_.get(), description_.c_str()),shout_.get(), "Failed to set description");
		shout_metadata_add(m, "description", description_.c_str());
	}
	if (!title_.empty()) {
		throw_shout_call(shout_set_name(shout_.get(), title_.c_str()),shout_.get(), "Failed to set title");
		shout_metadata_add(m, "title", title_.c_str());
	}




	throw_shout_call(shout_set_metadata(shout_.get(), m),shout_.get(), "Failed to set metadata");
	shout_metadata_free(m);
	throw_shout_call(shout_open(shout_.get()),shout_.get(),"Failed to open connection");

	log[log::info] << "Connection opened";


}

ShoutOutput::~ShoutOutput() noexcept
{
}

core::pFrame ShoutOutput::do_special_single_step(core::pCompressedVideoFrame frame)
{
	if (shout_send(shout_.get(), frame->data(), frame->size())!=SHOUTERR_SUCCESS) {
		log[log::warning] << "Failed to submit data to shoutcast server";
	}
	if (sync_) shout_sync(shout_.get());
	return {};
}
bool ShoutOutput::set_param(const core::Parameter& param)
{
	if (param.get_name() == "server") {
		server_ = param.get<std::string>();
	} else if (param.get_name() == "mount") {
		mount_ = param.get<std::string>();
	} else if (param.get_name() == "user") {
		user_ = param.get<std::string>();
	} else if (param.get_name() == "password") {
		password_ = param.get<std::string>();
	} else if (param.get_name() == "port") {
		port_ = param.get<uint16_t>();
	} else if (param.get_name() == "agent") {
		agent_ = param.get<std::string>();
	} else if (param.get_name() == "description") {
		description_ = param.get<std::string>();
	} else if (param.get_name() == "title") {
		title_ = param.get<std::string>();
	} else if (param.get_name() == "protocol") {
		auto it = shout_protocol_str.find(param.get<std::string>());
		if (it != shout_protocol_str.end()) {
			protocol_ = it->second;
		} else {
			log[log::warning] << "Unknown protocol " << param.get<std::string>();
			return false;
		}
	} else if (param.get_name() == "sync") {
		sync_ = param.get<bool>();
	} else if (param.get_name() == "url") {
		url_ = param.get<std::string>();
	} else return base_type::set_param(param);

	return true;
}

} /* namespace shout_cast */
} /* namespace yuri */
