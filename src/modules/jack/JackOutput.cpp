/*!
 * @file 		JackOutput.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		19.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "JackOutput.h"
#include "yuri/core/Module.h"
#include <cstdlib>

namespace yuri {
namespace jack_output {


IOTHREAD_GENERATOR(JackOutput)

MODULE_REGISTRATION_BEGIN("jack_output")
		REGISTER_IOTHREAD("jack_output",JackOutput)
MODULE_REGISTRATION_END()

core::Parameters JackOutput::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("JackOutput");
	p["channels"]["Number of output channels"]=2;
	p["allow_different_frequencies"]["Ignore sampling frequency from input frames"]=false;
	p["connect_to"]["Specify where to connect the outputs to (e.g. 'system')"]="";
	p["buffer_size"]["Size of internal buffer"]=1048576;
	return p;
}


namespace {
std::map<jack_status_t, std::string> error_codes = {
		{JackFailure, "Overall operation failed."},
		{JackInvalidOption,"The operation contained an invalid or unsupported option."},
		{JackNameNotUnique,"The desired client name was not unique. With the JackUseExactName option this situation is fatal. Otherwise, the name was modified by appending a dash and a two-digit number in the range \"-01\" to \"-99\". The jack_get_client_name() function will return the exact string that was used. If the specified client_name plus these extra characters would be too long, the open fails instead."},
		{JackServerStarted,"The JACK server was started as a result of this operation. Otherwise, it was running already. In either case the caller is now connected to jackd, so there is no race condition. When the server shuts down, the client will find out."},
		{JackServerFailed,"Unable to connect to the JACK server."},
		{JackServerError, "Communication error with the JACK server."},
		{JackNoSuchClient, "Requested client does not exist."},
		{JackLoadFailure,"Unable to load internal client"},
		{JackInitFailure,"Unable to initialize client"},
		{JackShmFailure,"Unable to access shared memory"},
		{JackVersionError,"Client's protocol version does not match"},
		{JackBackendError,"JackBackendError"},
		{JackClientZombie, "JackClientZombie"}
};

std::string get_error_string(jack_status_t err)
{
	if (!err) return "No error";
	auto it = error_codes.find(err);
	if (it != error_codes.end()) return it->second;

	std::string msg;
	for (const auto& e: error_codes) {
		if (err & e.first) msg += e.second + ", ";
	}
	if (!msg.empty()) {
		return msg;
	}
	return "Unknown";
}

int process_audio_wrapper(jack_nframes_t nframes, void *arg)
{
	auto j = reinterpret_cast<JackOutput*>(arg);
	return j->process_audio(nframes);
}

}


JackOutput::JackOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("jack_output")),handle_(nullptr),
client_name_("yuri_jack"),channels_(2),allow_different_frequencies_(false),buffer_size_(1048576)
{
	IOTHREAD_INIT(parameters)
	if (channels_ < 1) {
		throw exception::InitializationFailed("Invalid number of channels");
	}
	jack_status_t status;
	handle_ = {jack_client_open(client_name_.c_str(), JackNullOption, &status),[](jack_client_t*p){if(p)jack_client_close(p);}};
	if (!handle_) throw exception::InitializationFailed("Failed to connect to JACK server: " + get_error_string(status));

	if (status & JackServerStarted) {
		log[log::info] << "Jack server was started";
	}
	if (status&JackNameNotUnique) {
		client_name_ = jack_get_client_name(handle_.get());
		log[log::warning] << "Client name wasn't unique, we got new name from server instead: '" << client_name_ << "'";
	}
	log[log::info] << "Connected to JACK server";

	buffers_.resize(channels_, buffer_t<jack_default_audio_sample_t>(buffer_size_));

	if (jack_set_process_callback (handle_.get(), process_audio_wrapper, this)  !=0) {
		log[log::error] << "Failed to set process callback!";
	}

	for (size_t i=0;i<channels_;++i) {
		std::string port_name = "output" + lexical_cast<std::string>(i);
		auto port = jack_port_register (handle_.get(), port_name.c_str(),  JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
		if (!port) {
			throw exception::InitializationFailed("Failed to allocate output port");
		}
		ports_.push_back({port,[&](jack_port_t*p){if(p)jack_port_unregister(handle_.get(),p);}});
		log[log::info] << "Opened port " << port_name;
	}

	if (jack_activate (handle_.get())!=0) {
		throw exception::InitializationFailed("Failed to allocate output port");
	}
	log[log::info] << "client activated";
	const char **ports = nullptr;
	if (connect_to_.empty()) {
		ports = jack_get_ports (handle_.get(), nullptr, nullptr, JackPortIsPhysical|JackPortIsInput);
	} else {
		ports = jack_get_ports (handle_.get(), connect_to_.c_str(), nullptr, JackPortIsInput);
	}
	if (!ports) {
		log[log::warning] << "No suitable output ports found";
	} else {
		for (size_t i=0;i<ports_.size();++i) {
			if (!ports[i]) break;
			if (jack_connect (handle_.get(), jack_port_name (ports_[i].get()), ports[i])) {
				log[log::warning] << "Failed connect to output port " << i;
			} else {
				log[log::info] << "Connected port " << jack_port_name (ports_[i].get()) << " to " << ports[i];
			}
		}
		jack_free (ports);
	}
}

JackOutput::~JackOutput() noexcept
{
}

core::pFrame JackOutput::do_special_single_step(const core::pRawAudioFrame& frame)
{
	jack_nframes_t sample_rate =  jack_get_sample_rate(handle_.get());
	if (sample_rate != frame->get_sampling_frequency() && !allow_different_frequencies_) {
		log[log::warning] << "Frame has different sampling rate than JACKd, ignoring";
		return {};
	}

	size_t nframes = frame->get_sample_count();
	const int16_t * in_frames = reinterpret_cast<const int16_t*>(frame->data());

	const size_t in_channels = frame->get_channel_count();
	const size_t copy_channels = std::min(in_channels, ports_.size());

	std::unique_lock<std::mutex> lock(data_mutex_);
	if (copy_channels < ports_.size()) {
		for (size_t c=0;c<(ports_.size()-copy_channels);++c) {
			buffers_[c+copy_channels].push_silence(nframes);
		}
	}
	for (size_t i = 0; i< nframes;++i) {
		for (size_t c=0;c<copy_channels;++c) {
			buffers_[c].push(static_cast<jack_default_audio_sample_t>(*(in_frames+c))/std::numeric_limits<int16_t>::max());
		}
		in_frames+=in_channels;
	}

	return {};
}

int JackOutput::process_audio(jack_nframes_t nframes)
{
	std::unique_lock<std::mutex> lock(data_mutex_);
	size_t copy_count = std::min<size_t>(buffers_[0].size(), nframes);
	for (size_t i=0;i<buffers_.size();++i) {
		if (!ports_[i]) continue;
		jack_default_audio_sample_t* data = reinterpret_cast<jack_default_audio_sample_t *>(jack_port_get_buffer (ports_[i].get(), nframes));
		buffers_[i].pop(data,copy_count);
	}
	return 0;
}
bool JackOutput::set_param(const core::Parameter& param)
{
	if (param.get_name() == "channels") {
		channels_ = param.get<size_t>();
	} else if (param.get_name() == "allow_different_frequencies") {
		allow_different_frequencies_ = param.get<bool>();
	} else if (param.get_name() == "connect_to") {
		connect_to_ = param.get<std::string>();
	} else if (param.get_name() == "buffer_size") {
		buffer_size_ = param.get<size_t>();
	} else return base_type::set_param(param);
	return true;
}

} /* namespace jack_output */
} /* namespace yuri */
