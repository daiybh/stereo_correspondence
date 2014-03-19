/*!
 * @file 		JackOutput.cpp
 * @author 		<Your name>
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

}


JackOutput::JackOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("jack_output")),handle_(nullptr),
port_(nullptr),client_name_("yuri_jack"),port_name_("output")
{
	IOTHREAD_INIT(parameters)
	jack_status_t status;
	handle_ = jack_client_open(client_name_.c_str(), JackNullOption, &status);
	if (!handle_) throw exception::InitializationFailed("Failed to connect to JACK server: " + get_error_string(status));

	if (status & JackServerStarted) {
		log[log::info] << "Jack server was started";
	}
	if (status&JackNameNotUnique) {
		client_name_ = jack_get_client_name(handle_);
		log[log::warning] << "Client name wasn't unique, we got new name from server instead: '" << client_name_ << "'";
	}
	log[log::info] << "Connected to JACK server";

	port_ = jack_port_register (handle_, port_name_.c_str(),  JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
	if (!port_) {
		jack_client_close(handle_);
		throw exception::InitializationFailed("Failed to allocate output port");
	}

	// SET CALLBACKS

	if (jack_activate (handle_)!=0) {
		jack_port_unregister(handle_,port_);
		jack_client_close(handle_);
		throw exception::InitializationFailed("Failed to allocate output port");
	}

	const char **ports = jack_get_ports (handle_, nullptr, nullptr, JackPortIsPhysical|JackPortIsInput);
	if (!ports) {
		log[log::warning] << "No physical output ports";
	} else {
		if (jack_connect (handle_, jack_port_name (port_), ports[0])) {
			log[log::warning] << "Failed connect to output port";
		}
		std::free (ports);
	}

}

JackOutput::~JackOutput() noexcept
{
	if (handle_) jack_client_close(handle_);
}

core::pFrame JackOutput::do_special_single_step(const core::pRawAudioFrame& frame)
{
	jack_default_audio_sample_t *out;
	size_t nframes = frame->get_sample_count();// * frame->get_channel_count();
	out = reinterpret_cast<jack_default_audio_sample_t *>(jack_port_get_buffer (port_, nframes));

	const int16_t * in_frames = reinterpret_cast<const int16_t*>(frame->data());

	for (size_t i = 0; i< nframes;++i) {
		*out++ = static_cast<jack_default_audio_sample_t>(*in_frames)/32768.0;
		in_frames+=frame->get_channel_count();
	}

//	int16_t
//	memcpy (out, in,
//			sizeof (jack_default_audio_sample_t) * nframes);
	return {};
}
bool JackOutput::set_param(const core::Parameter& param)
{
	return base_type::set_param(param);
}

} /* namespace jack_output */
} /* namespace yuri */
