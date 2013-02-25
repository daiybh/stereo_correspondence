/*
 * DeckLinkInput.h
 *
 *  Created on: Sep 20, 2011
 *      Author: worker
 */

#ifndef DECKLINKINPUT_H_
#define DECKLINKINPUT_H_

#include "yuri/decklink/DeckLinkBase.h"

namespace yuri {

namespace decklink {

class DeckLinkInput:public DeckLinkBase, public IDeckLinkInputCallback {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

	DeckLinkInput(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~DeckLinkInput();

    virtual HRESULT VideoInputFormatChanged (BMDVideoInputFormatChangedEvents notificationEvents, IDeckLinkDisplayMode *newDisplayMode, BMDDetectedVideoInputFormatFlags detectedSignalFlags);
    virtual HRESULT VideoInputFrameArrived (IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket* audioPacket);


	virtual HRESULT STDMETHODCALLTYPE	QueryInterface (REFIID iid, LPVOID *ppv)	{return E_NOINTERFACE;}
	virtual ULONG STDMETHODCALLTYPE		AddRef ()									{return 1;}
	virtual ULONG STDMETHODCALLTYPE		Release ()									{return 1;}
	void run();
	bool init();
	bool step();
	bool start_capture();
	bool verify_display_mode();
	bool set_param(const core::Parameter &p);
private:
	IDeckLinkInput* input;
	yuri::uint_t width,height;
	BMDTimeValue value;
	BMDTimeScale scale;
	mutex schedule_mutex;
	bool detect_format;
	yuri::uint_t manual_detect_format;
	yuri::uint_t manual_detect_timeout;
	bool capture_stereo;
	bool disable_ntsc;
	bool disable_pal;
	bool disable_interlaced;
	bool disable_progressive;
	yuri::sshort_t audio_pipe;
	std::string current_format_name_;

	BMDDisplayMode select_next_format();
	bool restart_streams();
};

}

}

#endif /* DECKLINKINPUT_H_ */
