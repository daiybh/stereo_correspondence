/*
 * DeckLinkBase.cpp
 *
 *  Created on: Sep 22, 2011
 *      Author: worker
 */

#include "DeckLinkBase.h"
#include <boost/assign.hpp>
#include <boost/algorithm/string.hpp>
#include "yuri/core/Module.h"
namespace yuri {

namespace decklink {

std::map<std::string, BMDDisplayMode, yuri::core::compare_insensitive>
DeckLinkBase::mode_strings = boost::assign::map_list_of<std::string,BMDDisplayMode>
// SD Modes
		("pal",			bmdModePAL)
		("ntsc",		bmdModeNTSC)
		("ntsc2398",	bmdModeNTSC2398)
		("ntscp",		bmdModeNTSCp)
		("palp",		bmdModePALp)
// Progressive HD 1080 modes
		("1080p2398",	bmdModeHD1080p2398)
		("1080p24",		bmdModeHD1080p24)
		("1080p25",		bmdModeHD1080p25)
		("1080p2997",	bmdModeHD1080p2997)
		("1080p30",		bmdModeHD1080p30)
		("1080p50",		bmdModeHD1080p50)
		("1080p5994",	bmdModeHD1080p5994)
		("1080p60",		bmdModeHD1080p6000)
// Progressive 720p modes
		("720p50",		bmdModeHD720p50)
		("720p5994",	bmdModeHD720p5994)
		("720p60",		bmdModeHD720p60)
// Interlaced HD 1080 Modes
		("1080i50",		bmdModeHD1080i50)
		("1080i5994",	bmdModeHD1080i5994)
		("1080i60",		bmdModeHD1080i6000)
// PsF modes
		("1080p24PsF",	bmdModeHD1080p24)
		("1080p2398PsF",bmdModeHD1080p2398)
// 2k Modes
		("2k24",		bmdMode2k24)
		("2k2398",		bmdMode2k2398)
		("2k25",		bmdMode2k25);

std::map<std::string, BMDPixelFormat, yuri::core::compare_insensitive>
DeckLinkBase::pixfmt_strings=boost::assign::map_list_of<std::string,BMDPixelFormat>
		("yuv", 		bmdFormat8BitYUV)
		("v210", 		bmdFormat10BitYUV)
		("argb",		bmdFormat8BitARGB)
		("bgra",		bmdFormat8BitBGRA)
		("r210",		bmdFormat10BitRGB);

std::map<std::string, BMDVideoConnection, yuri::core::compare_insensitive>
DeckLinkBase::connection_strings=boost::assign::map_list_of<std::string, BMDVideoConnection>
		("SDI", 		bmdVideoConnectionSDI)
	    ("HDMI",		bmdVideoConnectionHDMI)
	    ("Optical SDI",	bmdVideoConnectionOpticalSDI)
	    ("Component",	bmdVideoConnectionComponent)
	    ("Composite",	bmdVideoConnectionComposite)
	    ("SVideo",		bmdVideoConnectionSVideo);

std::map<BMDPixelFormat, yuri::format_t>
DeckLinkBase::pixel_format_map=boost::assign::map_list_of<BMDPixelFormat, yuri::format_t>
		(bmdFormat8BitYUV,	YURI_FMT_UYVY422)
		(bmdFormat10BitYUV,	YURI_FMT_V210);
//		(bmdFormat8BitARGB)
//		(bmdFormat8BitBGRA)
//		(bmdFormat10BitRGB)
;

std::map<std::string, std::string>
progresive_to_psf=boost::assign::map_list_of<std::string, std::string>
("1080p24", "1080p24PsF")
("1080p2398", "1080p238PsF");
core::pParameters DeckLinkBase::configure()
{
	core::pParameters p = BasicIOThread::configure();
	p->set_description("DeckLink SDK Base");
	(*p)["device"]["Index of device to use"]=0;
	(*p)["format"]["Format"]="1080p25";
	(*p)["audio"]["Enable audio"]=false;
	(*p)["pixel_format"]["Select pixel format. Possible values are: (yuv, v210, argb, bgra, r210)"]="yuv";
	(*p)["audio_channels"]["Number of audio channels to process"]=2;
	return p;
}

BMDDisplayMode DeckLinkBase::parse_format(std::string fmt)
{
	boost::algorithm::to_lower(fmt);
	if (mode_strings.count(fmt)) return mode_strings[fmt];

	return bmdModeUnknown;
}

BMDVideoConnection DeckLinkBase::parse_connection(std::string fmt)
{
	if (connection_strings.count(fmt))
		return connection_strings[fmt];
	/*if (iequals(fmt,"SDI")) {
		return bmdVideoConnectionSDI;
	} else if (iequals(fmt,"HDMI")) {
		return bmdVideoConnectionHDMI;
	} else if (iequals(fmt,"OpticalSDI")) {
		return bmdVideoConnectionOpticalSDI;
	} else if (iequals(fmt,"Component")) {
		return bmdVideoConnectionComponent;
	} else if (iequals(fmt,"Composite")) {
		return bmdVideoConnectionComposite;
	} else if (iequals(fmt,"SVIDEO")) {
		return bmdVideoConnectionSVideo;
	} */
	else return bmdVideoConnectionHDMI;
}

const std::string DeckLinkBase::bmerr(HRESULT res)
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

DeckLinkBase::DeckLinkBase(log::Log &log_, core::pwThreadBase parent, yuri::sint_t inp, yuri::sint_t outp, core::Parameters &parameters, std::string name)
	:BasicIOThread(log_,parent,inp,outp,name),device(0),device_index(0),connection(bmdVideoConnectionHDMI),
	 mode(bmdModeHD1080p25),pixel_format(bmdFormat8BitYUV),
	 audio_sample_rate(bmdAudioSampleRate48kHz),audio_sample_type(bmdAudioSampleType16bitInteger),
	 audio_channels(2),audio_enabled(false),actual_format_is_psf(false)
{

}

DeckLinkBase::~DeckLinkBase()
{

}

bool DeckLinkBase::set_param(const core::Parameter &p)
{
	using boost::iequals;
	if (iequals(p.name, "device")) {
		device_index = p.get<yuri::ushort_t>();
	} else if (iequals(p.name, "connection")) {
		connection=parse_connection(p.get<std::string>());
	} else if (iequals(p.name, "format")) {
		BMDDisplayMode m = parse_format(p.get<std::string>());
		if (m == bmdModeUnknown) {
			mode = bmdModeHD1080p25;
			log[log::error] << "Failed to parse format, falling back "
					"to 1080p25\n";
			actual_format_is_psf = false;
		} else {
			mode = m;
			actual_format_is_psf = is_psf(p.get<std::string>());
		}

	} else if (iequals(p.name, "audio")) {
		audio_enabled=p.get<bool>();
	} else if (iequals(p.name, "pixel_format")) {
		if (pixfmt_strings.count(p.get<std::string>())) {
			pixel_format=pixfmt_strings[p.get<std::string>()];
			log[log::debug] << "Pixelformat set to " << p.get<std::string>() << std::endl;
		}
		else {
			log[log::warning] << "Unknown pixel format specified (" << p.get<std::string>()
			<< ". Using default 8bit YUV 4:2:2." << std::endl;
			pixel_format=bmdFormat8BitYUV;
		}
	} else if (iequals(p.name, "audio_channels")) {
		audio_channels=p.get<size_t>();
	} else return BasicIOThread::set_param(p);
	return true;
}

bool DeckLinkBase::init_decklink()
{
	IDeckLinkIterator *iter = CreateDeckLinkIteratorInstance();
	if (!iter) return false;
	yuri::ushort_t idx = 0;
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
	for (std::map<std::string, BMDDisplayMode, yuri::core::compare_insensitive>::iterator it=mode_strings.begin();
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
bool DeckLinkBase::is_psf(std::string name)
{
	typedef std::map<std::string, std::string>::iterator miter;
	for (miter it=progresive_to_psf.begin();it!=progresive_to_psf.end();++it) {
		if (it->second == name) return true;
	}
	return false;
}
}

}
