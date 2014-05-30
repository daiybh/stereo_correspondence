/*!
 * @file 		JackInput.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		19.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "JackInput.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include <cstdlib>

namespace yuri {
namespace jack {


IOTHREAD_GENERATOR(JackInput)

core::Parameters JackInput::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("JackInput");
	p["channels"]["Number of output channels"]=2;
	p["connect_to"]["Specify where to connect the outputs to (e.g. 'system')"]="";
	p["client_name"]["Name of the JACK client"]="yuri";
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
	auto j = reinterpret_cast<JackInput*>(arg);
	return j->process_audio(nframes);
}

}


JackInput::JackInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("jack_output")),handle_(nullptr),
client_name_("yuri_jack"),channels_(2),sample_rate_(48000)

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

	//buffers_.resize(channels_, buffer_t<jack_default_audio_sample_t>(buffer_size_));

	if (jack_set_process_callback (handle_.get(), process_audio_wrapper, this)  !=0) {
		log[log::error] << "Failed to set process callback!";
	}

	for (size_t i=0;i<channels_;++i) {
		std::string port_name = "input" + lexical_cast<std::string>(i);
		auto port = jack_port_register (handle_.get(), port_name.c_str(),  JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
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
		ports = jack_get_ports (handle_.get(), nullptr, nullptr, JackPortIsPhysical|JackPortIsOutput);
	} else {
		ports = jack_get_ports (handle_.get(), connect_to_.c_str(), nullptr, JackPortIsOutput);
	}
	if (!ports) {
		log[log::warning] << "No suitable output ports found";
	} else {
		for (size_t i=0;i<ports_.size();++i) {
			if (!ports[i]) break;
			if (jack_connect (handle_.get(), ports[i], jack_port_name (ports_[i].get()))) {
				log[log::warning] << "Failed connect to output port " << i;
			} else {
				log[log::info] << "Connected port " << jack_port_name (ports_[i].get()) << " to " << ports[i];
			}
		}
		jack_free (ports);
	}
	sample_rate_ = jack_get_sample_rate(handle_.get());
	log[log::info] << "Using sample rate " << sample_rate_ << "Hz";
	//using namespace core::raw_audio_format;
	//set_supported_formats({unsigned_8bit, unsigned_16bit, unsigned_32bit, signed_16bit, signed_32bit, float_32bit});
}

JackInput::~JackInput() noexcept
{
}

core::pFrame JackInput::do_special_single_step(const core::pRawAudioFrame& /*frame*/)
{
	return {};
}

int JackInput::process_audio(jack_nframes_t nframes)
{
	std::unique_lock<std::mutex> lock(data_mutex_);
//	log[log::info] << "Received " << nframes;
	// Querying sample rate again in a case it would change (can it ever happen?)
	sample_rate_ = jack_get_sample_rate(handle_.get());
	auto frame = core::RawAudioFrame::create_empty(core::raw_audio_format::float_32bit,
					channels_, sample_rate_, nframes);
	float* data = reinterpret_cast<float*>(frame->data());
	for (size_t i=0;i<channels_ /* buffers_.size()*/;++i) {
		float* data_ptr = data + i;
		if (!ports_[i]) continue;
		jack_default_audio_sample_t* data = reinterpret_cast<jack_default_audio_sample_t *>(jack_port_get_buffer (ports_[i].get(), nframes));
		for (jack_nframes_t i=0;i<nframes;++i) {
			*data_ptr = *data++;
			data_ptr+=channels_;
		}
	}
	push_frame(0, frame);
	return 0;
}
bool JackInput::set_param(const core::Parameter& param)
{
	if (param.get_name() == "channels") {
		channels_ = param.get<size_t>();
	} else if (param.get_name() == "connect_to") {
		connect_to_ = param.get<std::string>();
	} else if (param.get_name() == "client_name") {
		client_name_ = param.get<std::string>();
	} else return base_type::set_param(param);
	return true;
}

} /* namespace jack_output */
} /* namespace yuri */
