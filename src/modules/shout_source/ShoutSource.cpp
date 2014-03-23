/*!
 * @file 		ShoutSource.cpp
 * @author 		<Your name>
 * @date		23.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "ShoutSource.h"
#include "yuri/core/Module.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/CompressedVideoFrame.h"

namespace yuri {
namespace shout_source {


IOTHREAD_GENERATOR(ShoutSource)

MODULE_REGISTRATION_BEGIN("shout_source")
		REGISTER_IOTHREAD("shout_source",ShoutSource)
MODULE_REGISTRATION_END()

core::Parameters ShoutSource::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("ShoutSource");
	p["port"]["Port to connect to"]=5900;
	p["address"]["Address to connect to"]="127.0.0.1";
	p["socket"]["Socket implementation"]="yuri_tcp";
	p["mount"]["Mountpoint"]="";
	return p;
}


ShoutSource::ShoutSource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("shout_source"))
{
	IOTHREAD_INIT(parameters)
	set_latency(10_ms);
	if (mount_.empty()) throw exception::InitializationFailed("No mountpoint specified");
	socket_ = core::StreamSocketGenerator::get_instance().generate(socket_impl_, log);
	log[log::info] << "Created socket";
	log[log::info] << "connecting to " << address_ << ":" << port_;
	if (!socket_->connect(address_,port_)) throw exception::InitializationFailed("Failed to connect to server");
}

ShoutSource::~ShoutSource() noexcept
{
}

namespace {
template<class Iter>
std::string get_string(Iter& begin, Iter end)
{
	std::string out;
	Iter b = begin;
	while (begin != end) {
		auto c =*begin;
		out+=c;
		if (c=='\n' || c=='\r') {
			break;
		}
		begin++;
	}
	if (*begin == '\n' || *begin == '\r') {
		begin++;
		if (*begin == '\n' || *begin=='\r') begin++;
	} else {
		begin = b;
		out = "";
	}
	return out;
}

std::string trim_string(const std::string& str)
{
	auto sub0 = str.find_first_not_of(" \n\r\t");
	auto sub1 = str.find_last_not_of(" \n\r\t");
	// We should trim the end as well...

	return str.substr(sub0, sub1 - sub0 +1);
}

std::pair<std::string, std::string>
split_line(const std::string& line) {
	auto index = line.find(":");
	if (index == line.npos) return {trim_string(line),""};
	return {trim_string(line.substr(0,index)), trim_string(line.substr(index+1))};
}

}

void ShoutSource::run()
{
	std::string request = "GET /" + mount_ + " HTTP/1.0\r\n\r\n";//Icy-MetaData:1\r\n\r\n";
	log[log::info] << "Requesting mount point " << mount_;
	socket_->send_data(request);

	std::vector<uint8_t> data(1024);
	size_t len;

	std::vector<char> data2;

	// parse header
	bool header_parsed = false;
	std::map<std::string, std::string> headers;
	while (!header_parsed && still_running()) {
		socket_->wait_for_data(get_latency());

		data.resize(1024);
		len = socket_->receive_data(data);
		if (len) {
			size_t data2_len= data2.size();
			data2.resize(data2_len + len);
			data.resize(len);
			std::copy(data.begin(),data.end(), data2.begin()+data2_len);
			auto iter = data2.begin();
			while (iter != data2.end()) {
				auto line = get_string(iter, data2.end());
				log[log::info] << "Line size " << line.size() << ", data2 size " <<data2.size();
				if (line.size()>1) {
					auto h = split_line(line);
					headers.insert(h);

				} else if (line.size() == 1) {
					log[log::info] << "END of header";
					header_parsed = true;
					break;
				} else {
					std::vector<char> d(iter, data2.end());
					data2.swap(d);
					break;
				}
			}
		}
	}

	log[log::info] << "Parsed headers:";
	for (const auto&h: headers) {
		log[log::info] << "\t" << h.first << " : '" << h.second << "'";
	}
	format_t format = core::compressed_frame::ogg;
	auto it = headers.find("Content-Type");
	if (it == headers.end()) {
		log[log::warning] << "No content type specified, assuming ogg";
	} else {
		format = core::compressed_frame::get_format_from_mime(it->second);
		if (!format) {
			log[log::warning] << "Unknown format received from server. Assuming it's ogg...";
			format = core::compressed_frame::ogg;
		} else {
			log[log::info] << "Using format: " << core::compressed_frame::get_format_name(format);
		}
	}

	while (still_running()) {
		socket_->wait_for_data(get_latency());
		data.resize(1024);
		len = socket_->receive_data(data);
		if (len) {
			data.resize(len);
			auto frame = core::CompressedVideoFrame::create_empty(core::compressed_frame::ogg, resolution_t{0,0}, data.data(), len);
			push_frame(0, frame);
		}
	}
}
bool ShoutSource::set_param(const core::Parameter& param)
{
	if (param.get_name() == "address") {
		address_ = param.get<std::string>();
	} else if (param.get_name() == "mount") {
		mount_ = param.get<std::string>();
	} else if (param.get_name() == "port") {
		port_=param.get<uint16_t>();
	} else if (param.get_name() == "socket") {
		socket_impl_ = param.get<std::string>();
	} return core::IOThread::set_param(param);
	return true;
}

} /* namespace shout_source */
} /* namespace yuri */

