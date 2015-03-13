/*!
 * @file 		DeckLinkBase.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		22.9.2011
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2011 - 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DECKLINKBASE_H_
#define DECKLINKBASE_H_
#include "yuri/core/thread/IOThread.h"
#include "DeckLinkAPI.h"
#include "yuri/core/thread/InputThread.h"

namespace yuri {

namespace decklink {

BMDDisplayMode parse_format(const std::string& fmt);
std::string bm_mode_to_yuri(BMDDisplayMode fmt);
BMDVideoConnection parse_connection(const std::string& fmt);
std::string bm_connection_to_yuri(BMDVideoConnection fmt);
const std::string bmerr(HRESULT res);
BMDPixelFormat convert_yuri_to_bm(format_t fmt);
format_t convert_bm_to_yuri(BMDPixelFormat fmt);
BMDDisplayMode get_next_format(BMDDisplayMode);
inline void decklink_deleter(IUnknown *ptr) { if (ptr) ptr->Release(); }

class DeckLinkBase:public core::IOThread {
public:
	static core::Parameters configure();
	static std::vector<core::InputDeviceInfo> enumerate_inputs();

	DeckLinkBase(const log::Log &log_, core::pwThreadBase parent, position_t inp, position_t outp, const std::string& name="DeckLink");
	virtual ~DeckLinkBase() noexcept;

	bool set_param(const core::Parameter &p);
	bool init_decklink();
protected:
	IDeckLink * device;
	uint16_t device_index;
	uint16_t connection;
	BMDDisplayMode mode;
	BMDPixelFormat pixel_format;
	BMDAudioSampleRate audio_sample_rate;
	BMDAudioSampleType audio_sample_type;
	unsigned audio_channels;
	bool audio_enabled;
	bool actual_format_is_psf;
//	static std::map<std::string, BMDDisplayMode, compare_insensitive> mode_strings;
//	static std::map<std::string, BMDPixelFormat, compare_insensitive> pixfmt_strings;
//	static std::map<std::string, BMDVideoConnection, compare_insensitive> connection_strings;
//	static std::map<BMDPixelFormat, yuri::format_t> pixel_format_map;
	static std::string get_mode_name(BMDDisplayMode, bool psf=false);
	static bool is_psf(const std::string& name);

};

}

}

#endif /* DECKLINKBASE_H_ */
