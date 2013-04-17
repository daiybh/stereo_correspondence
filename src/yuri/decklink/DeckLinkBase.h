/*
 * DeckLinkBase.h
 *
 *  Created on: Sep 22, 2011
 *      Author: worker
 */

#ifndef DECKLINKBASE_H_
#define DECKLINKBASE_H_
#include "yuri/core/BasicIOThread.h"
#include "DeckLinkAPI.h"
#include "yuri/core/BasicPipe.h"
namespace yuri {

namespace decklink {

class DeckLinkBase:public core::BasicIOThread {
public:
	static core::pParameters configure();
	static BMDDisplayMode parse_format(std::string fmt);
	static BMDVideoConnection parse_connection(std::string fmt);
	static const std::string bmerr(HRESULT res);

	DeckLinkBase(log::Log &log_, core::pwThreadBase parent, yuri::sint_t inp, yuri::sint_t outp, core::Parameters &parameters, std::string name="DeckLink");
	virtual ~DeckLinkBase();

	bool set_param(const core::Parameter &p);
	bool init_decklink();
	static void decklink_deleter(IUnknown *ptr) { if (ptr) ptr->Release(); }
protected:
	IDeckLink * device;
	yuri::ushort_t device_index;
	yuri::ushort_t connection;
	BMDDisplayMode mode;
	BMDPixelFormat pixel_format;
	BMDAudioSampleRate audio_sample_rate;
	BMDAudioSampleType audio_sample_type;
	yuri::uint_t audio_channels;
	bool audio_enabled;
	bool actual_format_is_psf;
	static std::map<std::string, BMDDisplayMode, yuri::core::compare_insensitive> mode_strings;
	static std::map<std::string, BMDPixelFormat, yuri::core::compare_insensitive> pixfmt_strings;
	static std::map<std::string, BMDVideoConnection, yuri::core::compare_insensitive> connection_strings;
	static std::map<BMDPixelFormat, yuri::format_t> pixel_format_map;
	static std::string get_mode_name(BMDDisplayMode, bool psf=false);
	static bool is_psf(std::string name);

};

}

}

#endif /* DECKLINKBASE_H_ */
