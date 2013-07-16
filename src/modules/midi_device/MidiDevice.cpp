/*!
 * @file 		MidiDevice.cpp
 * @author 		<Your name>
 * @date		12.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "MidiDevice.h"
#include "yuri/core/Module.h"
#include <poll.h>

namespace yuri {
namespace midi_device {

REGISTER("midi_device",MidiDevice)

IO_THREAD_GENERATOR(MidiDevice)

core::pParameters MidiDevice::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("MidiDevice");
	(*p)["device"]["Device name. Keep the default value, unless you know what you're doing."]=std::string("default");
	(*p)["connection"]["Output port to connect to. It could be either in form client:port, or a name of the port."]=std::string("");
	p->set_max_pipes(0,0);
	return p;
}


MidiDevice::MidiDevice(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,0,0,std::string("midi_device")),
event::BasicEventProducer(log),
sequencer_(nullptr), device_("default"),connection_("")
{
	IO_THREAD_INIT("midi_device")
	latency = 1000;
	int error;

	if ((error = snd_seq_open(&sequencer_, device_.c_str(), SND_SEQ_OPEN_INPUT, SND_SEQ_NONBLOCK)) < 0) {
		log[log::error] << "Failed to open device " << device_ << ", error: " << error;
		throw exception::InitializationFailed("Failed to open device");
	}
	snd_seq_set_client_name(sequencer_, "Yuri MIDI input");
	port_ = snd_seq_create_simple_port(sequencer_, get_node_name().c_str()/*"listen:in"*/,
					  SND_SEQ_PORT_CAP_WRITE|SND_SEQ_PORT_CAP_SUBS_WRITE,
					  SND_SEQ_PORT_TYPE_APPLICATION);
	if (!connection_.empty()) {
		snd_seq_addr_t src;
		if (snd_seq_parse_address(sequencer_, &src, connection_.c_str()) < 0) {
			log[log::warning] << "Failed to get address of device '" << connection_ <<"'";
		} else {
			log[log::info] << "Found address of device " << connection_ << ": "
								<< static_cast<int>(src.client) << ":"
								<< static_cast<int>(src.port);
			snd_seq_addr_t dest = {	static_cast<unsigned char>(snd_seq_client_id(sequencer_)),
									static_cast<unsigned char>(port_)};
			snd_seq_port_subscribe_t* subs;
			snd_seq_port_subscribe_alloca(&subs);
			snd_seq_port_subscribe_set_sender(subs, &src);
			snd_seq_port_subscribe_set_dest(subs, &dest);
//			snd_seq_port_subscribe_set_queue(subs, 0);
//			snd_seq_port_subscribe_set_exclusive(subs, 0);
//			snd_seq_port_subscribe_set_time_update(subs, 0);
//			snd_seq_port_subscribe_set_time_real(subs, 0);
			if (snd_seq_subscribe_port(sequencer_, subs) < 0) {
				log[log::warning] << "Failed to subscribe to device " << connection_;
			} else {
				log[log::info] << "Succesfully subscribed to device " << connection_;
			}
		}
	}
//	snd_seq_drop_input(sequencer_);
	snd_seq_drop_input_buffer(sequencer_);
}

MidiDevice::~MidiDevice()
{
}

bool MidiDevice::process_event(const snd_seq_event_t& midievent)
{
	std::string name;
	switch(midievent.type) {
		case SND_SEQ_EVENT_CONTROLLER:
			name = std::string("control_")
					+ lexical_cast<std::string>(static_cast<int>(midievent.data.control.channel))
					+ "_" + lexical_cast<std::string>(midievent.data.control.param);
			log[log::debug] << "(" << name << ") Controller, channel "
			<< static_cast<int>(midievent.data.control.channel)
			<< ", param: " << midievent.data.control.param
			<< ", value: " << midievent.data.control.value;
			emit_event(name,make_shared<event::EventInt>(midievent.data.control.value,0,127));
			break;
		case SND_SEQ_EVENT_NOTEON:
			name = std::string("note_")
					+ lexical_cast<std::string>(static_cast<int>(midievent.data.note.channel))
					+ "_" + lexical_cast<std::string>(static_cast<int>(midievent.data.note.note));
			log[log::debug] << "(" << name << ") NOTE ON" //, channel "
//			<< static_cast<int>(midievent.data.note.channel)
//			<< ", param: " << midievent.data.note.note
			<< ", velocity: " << static_cast<int>(midievent.data.note.velocity);
			emit_event(name,make_shared<event::EventBool>(true));
			break;
		case SND_SEQ_EVENT_NOTEOFF:
			name = std::string("note_")
								+ lexical_cast<std::string>(static_cast<int>(midievent.data.note.channel))
								+ "_" + lexical_cast<std::string>(static_cast<int>(midievent.data.note.note));
			log[log::debug] << "Note OFF"
				<< ", velocity: " << static_cast<int>(midievent.data.note.velocity);
			emit_event(name,make_shared<event::EventBool>(false));
			break;
	}
	return true;
}
void MidiDevice::run()
{
	IO_THREAD_PRE_RUN
	int npfd;

	npfd = snd_seq_poll_descriptors_count(sequencer_, POLLIN);
	std::vector<pollfd> pfd(npfd);
	snd_seq_poll_descriptors(sequencer_, &pfd[0], npfd, POLLIN);
	snd_seq_event_t *midievent;
	while(still_running())
	{
		if (poll(&pfd[0], npfd, latency/1000) > 0) {
			do {
				snd_seq_event_input(sequencer_, &midievent);
				if (midievent) {
					process_event(*midievent);
				}
			} while (snd_seq_event_input_pending(sequencer_, 0) > 0);
		}
	}
	IO_THREAD_POST_RUN
}
bool MidiDevice::set_param(const core::Parameter& param)
{
	if (iequals(param.name, "device")) {
		device_ = param.get<std::string>();
	} else if (iequals(param.name, "connection")) {
		connection_ = param.get<std::string>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace midi_device */
} /* namespace yuri */
