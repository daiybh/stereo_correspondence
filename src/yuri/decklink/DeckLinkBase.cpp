/*
 * DeckLinkBase.cpp
 *
 *  Created on: Sep 22, 2011
 *      Author: worker
 */

#include "DeckLinkBase.h"
//#include <boost/assign.hpp>
//#include <boost/algorithm/string.hpp>
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/utils.h"
namespace yuri {

namespace decklink {

namespace {
using namespace yuri::core::raw_format;
struct compare_insensitive
{
	bool operator()(const std::string& a, const std::string& b) const
	{
		return iequals(a,b);
	}
};

std::map<std::string, BMDDisplayMode, compare_insensitive>
mode_strings = {
// SD Modes
		{"pal",			bmdModePAL},
		{"ntsc",		bmdModeNTSC},
		{"ntsc2398",	bmdModeNTSC2398},
		{"ntscp",		bmdModeNTSCp},
		{"palp",		bmdModePALp},
// Progressive HD 1080 modes
		{"1080p2398",	bmdModeHD1080p2398},
		{"1080p24",		bmdModeHD1080p24},
		{"1080p25",		bmdModeHD1080p25},
		{"1080p2997",	bmdModeHD1080p2997},
		{"1080p30",		bmdModeHD1080p30},
		{"1080p50",		bmdModeHD1080p50},
		{"1080p5994",	bmdModeHD1080p5994},
		{"1080p60",		bmdModeHD1080p6000},
// Progressive 720p modes
		{"720p50",		bmdModeHD720p50},
		{"720p5994",	bmdModeHD720p5994},
		{"720p60",		bmdModeHD720p60},
// Interlaced HD 1080 Modes
		{"1080i50",		bmdModeHD1080i50},
		{"1080i5994",	bmdModeHD1080i5994},
		{"1080i60",		bmdModeHD1080i6000},
// PsF modes
		{"1080p24PsF",	bmdModeHD1080p24},
		{"1080p2398PsF",bmdModeHD1080p2398},
// 2k Modes
		{"2k24",		bmdMode2k24},
		{"2k2398",		bmdMode2k2398},
		{"2k25",		bmdMode2k25},
};

std::map<std::string, BMDPixelFormat, compare_insensitive>
pixfmt_strings = {
		{"yuv", 		bmdFormat8BitYUV},
		{"v210", 		bmdFormat10BitYUV},
		{"argb",		bmdFormat8BitARGB},
		{"bgra",		bmdFormat8BitBGRA},
		{"r210",		bmdFormat10BitRGB},
};

std::map<std::string, BMDVideoConnection, compare_insensitive>
connection_strings = {
		{"SDI", 		bmdVideoConnectionSDI},
	    {"HDMI",		bmdVideoConnectionHDMI},
	    {"Optical SDI",	bmdVideoConnectionOpticalSDI},
	    {"Component",	bmdVideoConnectionComponent},
	    {"Composite",	bmdVideoConnectionComposite},
	    {"SVideo",		bmdVideoConnectionSVideo},
};

std::map<BMDPixelFormat, yuri::format_t> pixel_format_map = {
		{bmdFormat8BitYUV,	uyvy422},
		{bmdFormat10BitYUV,	yuv422_v210},
		{bmdFormat8BitARGB,	argb32},
		{bmdFormat8BitBGRA,	bgra32},
		{bmdFormat10BitRGB,	rgb_r10k},
};

std::map<std::string, std::string>
progresive_to_psf = {
{"1080p24", "1080p24PsF"},
{"1080p2398", "1080p2398PsF"},
};
}
core::Parameters DeckLinkBase::configure()
{
	core::Parameters p = IOThread::configure();
	p.set_description("DeckLink SDK Base");
	p["device"]["Index of device to use"]=0;
	p["format"]["Format"]="1080p25";
	p["audio"]["Enable audio"]=false;
	p["pixel_format"]["Select pixel format. Possible values are: (yuv, v210, argb, bgra, r210)"]="yuv";
	p["audio_channels"]["Number of audio channels to process"]=2;
	return p;
}

BMDDisplayMode parse_format(const std::string& fmt)
{
//	boost::algorithm::to_lower(fmt);
	if (mode_strings.count(fmt)) return mode_strings[fmt];

	return bmdModeUnknown;
}

BMDVideoConnection parse_connection(const std::string& fmt)
{
	if (connection_strings.count(fmt))
		return connection_strings[fmt];
	else return bmdVideoConnectionHDMI;
}
BMDPixelFormat convert_yuri_to_bm(format_t fmt)
{
	for (auto f: pixel_format_map) {
		if (f.second == fmt) return f.first;
	}
	return 0;
}
format_t convert_bm_to_yuri(BMDPixelFormat fmt)
{
	auto it = pixel_format_map.find(fmt);
	if (it == pixel_format_map.end()) return 0;
	return it->second;
}
const std::string bmerr(HRESULT res)
{
	switch (HRESULT_CODE(res)) {
		case S_OK: return "OK";
		case S_FALSE: return "False";
	}
	switch (res) {
		case E_FAIL: return "Failed";
		case E_ACCESSDENIED: return "Access denied";
		case E_OUTOFMEMORY: return "OUTOFMEMORY";
		case E_INVALIDARG:return "Invalid argument";
		default: return "Unknown";
	}
}
BMDDisplayMode get_next_format(BMDDisplayMode fmt)
{
	auto it = mode_strings.begin();
	for (; it != mode_strings.end(); ++it) {
		if (it->second == fmt) {
			++it;break;
		}
	}
	if (it == mode_strings.end())
		return mode_strings.begin()->second;
	return it->second;
}
DeckLinkBase::DeckLinkBase(const log::Log &log_, core::pwThreadBase parent, position_t inp, position_t outp, const std::string& name)
	:IOThread(log_,parent,inp,outp,name),device(0),device_index(0),connection(bmdVideoConnectionHDMI),
	 mode(bmdModeHD1080p25),pixel_format(bmdFormat8BitYUV),
	 audio_sample_rate(bmdAudioSampleRate48kHz),audio_sample_type(bmdAudioSampleType16bitInteger),
	 audio_channels(2),audio_enabled(false),actual_format_is_psf(false)
{

}

DeckLinkBase::~DeckLinkBase() noexcept
{

}

bool DeckLinkBase::set_param(const core::Parameter &p)
{
	if (iequals(p.get_name(), "device")) {
		device_index = p.get<uint16_t>();
	} else if (iequals(p.get_name(), "connection")) {
		connection=parse_connection(p.get<std::string>());
	} else if (iequals(p.get_name(), "format")) {
		BMDDisplayMode m = parse_format(p.get<std::string>());
		if (m == bmdModeUnknown) {
			mode = bmdModeHD1080p25;
			log[log::error] << "Failed to parse format, falling back "
					"to 1080p25";
			actual_format_is_psf = false;
		} else {
			mode = m;
			actual_format_is_psf = is_psf(p.get<std::string>());
		}

	} else if (iequals(p.get_name(), "audio")) {
		audio_enabled=p.get<bool>();
	} else if (iequals(p.get_name(), "pixel_format")) {
		if (pixfmt_strings.count(p.get<std::string>())) {
			pixel_format=pixfmt_strings[p.get<std::string>()];
			log[log::debug] << "Pixelformat set to " << p.get<std::string>();
		}
		else {
			log[log::warning] << "Unknown pixel format specified (" << p.get<std::string>()
			<< ". Using default 8bit YUV 4:2:2.";
			pixel_format=bmdFormat8BitYUV;
		}
	} else if (iequals(p.get_name(), "audio_channels")) {
		audio_channels=p.get<size_t>();
	} else return IOThread::set_param(p);
	return true;
}

bool DeckLinkBase::init_decklink()
{
	IDeckLinkIterator *iter = CreateDeckLinkIteratorInstance();
	if (!iter) return false;
	uint16_t idx = 0;
	while (iter->Next(&device)==S_OK) {
		//if (!device) continue;
		if (idx==device_index) {
			break;
		}
		if (!device) {
			log[log::warning] << "Did not receive valid device for index " << idx << std::endl;
		} else {
			device->Release();
			device = 0;
			log[log::debug] << "Skipping device " << idx << std::endl;
		}
		idx++;
	}
	if (device_index != idx) {
		log[log::fatal] << "There is no device with index " << device_index << " connected to he system." << std::endl;
		return false;
	}
	if (!device) {
		log[log::fatal] << "Device not allocated properly!" << std::endl;
		return false;
	}
	const char * device_name = 0;
	device->GetDisplayName(&device_name);
	if (device_name) {
		log[log::info] << "Using blackmagic device: " << device_name << std::endl;
		delete device_name;
	}
	return true;
}
std::string DeckLinkBase::get_mode_name(BMDDisplayMode mode, bool psf)
{
	for (auto it=mode_strings.begin();
			it!=mode_strings.end();++it) {
		if (it->second == mode) {
			if (psf) {
				if (progresive_to_psf.count(it->first)) {
					return progresive_to_psf[it->first];
				}
			}
			return it->first;
		}
	}
	return std::string();
}
bool DeckLinkBase::is_psf(const std::string& name)
{
	for (auto it=progresive_to_psf.begin();it!=progresive_to_psf.end();++it) {
		if (it->second == name) return true;
	}
	return false;
}
}

}
