/*!
 * @file 		MidiDevice.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		12.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef MIDIDEVICE_H_
#define MIDIDEVICE_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventProducer.h"
#include <alsa/asoundlib.h>

namespace yuri {
namespace midi_device {

class MidiDevice: public core::IOThread, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	MidiDevice(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~MidiDevice() noexcept;
private:
	virtual void run();

	virtual bool set_param(const core::Parameter& param);
	bool process_event(const snd_seq_event_t& midievent);
	snd_seq_t*			sequencer_;
	std::string			device_;
	int 				port_;
	std::string			connection_;
};

} /* namespace midi_device */
} /* namespace yuri */
#endif /* MIDIDEVICE_H_ */
