/*
 * DeckLinkInput.cpp
 *
 *  Created on: Sep 20, 2011
 *      Author: worker
 */

#include "DeckLinkInput.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace decklink {

REGISTER("decklink_input",DeckLinkInput)


IO_THREAD_GENERATOR(DeckLinkInput)


core::pParameters DeckLinkInput::configure()
{
	core::pParameters p = DeckLinkBase::configure();
	p->set_description("Outputs input video to BlackMagic Design device (Eclipse, Intensity, ...)");
	(*p)["format"]["Output format (1080p25, etc)"]="1080p25";
	(*p)["connection"]["Output connection (HDMI, SDI, SVideo, ...). Please note that enabling one output will also enable other compatible outputs"]=std::string("HDMI");
	(*p)["device"]["Index of device to use"]=0;
	(*p)["format_detection"]["Try to autodetect video format."]=1;
	(*p)["force_detection"]["Force autodetecting video format. EXPERIMENTAL"]=0;
	(*p)["stereo"]["Capture duallink stereo."]=false;
	(*p)["disable_ntsc"]["Disable NTSC modes."]=false;
	(*p)["disable_pal"]["Disable PAL modes."]=false;
	(*p)["disable_interlaced"]["Disable interlaced modes."]=false;
	(*p)["disable_progressive"]["Disable progressive modes."]=false;
	return p;
}

DeckLinkInput::DeckLinkInput(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		DeckLinkBase(log_,parent,0,1,parameters,"DeckLinkInput"),
		detect_format(true),manual_detect_format(0),manual_detect_timeout(0),
		capture_stereo(false),disable_ntsc(false),disable_pal(true),disable_interlaced(false),
		disable_progressive(false),audio_pipe(-1)
{
	IO_THREAD_INIT("Decklink Input")
	resize(0,1+(capture_stereo?1:0)+(audio_enabled?1:0));
	if (audio_enabled) audio_pipe=capture_stereo?2:1;
	current_format_name_ = get_mode_name(mode);
	if (!init()) {
		throw exception::InitializationFailed("Failed to initialize Decklink Input");
	}
}

DeckLinkInput::~DeckLinkInput()
{
	if (input) input->Release();
	if (device) device->Release();
}

HRESULT DeckLinkInput::VideoInputFormatChanged (BMDVideoInputFormatChangedEvents notificationEvents, IDeckLinkDisplayMode *newDisplayMode, BMDDetectedVideoInputFormatFlags detectedSignalFlags)
{
	log[log::debug] << "VideoInputFormatChanged" << "\n";
	return S_OK;
}
HRESULT DeckLinkInput::VideoInputFrameArrived (IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket* audioPacket)
{
	using std::swap;
	int res = S_OK;
	log[log::debug] << "VideoInputFrameArrived" << "\n";
	if (videoFrame->GetFlags()&bmdFrameHasNoInputSource) {
		//LOG(log,warning,"No input detected");
		log[log::warning] << "No input detected" << "\n";
		if (manual_detect_format && ++manual_detect_timeout>manual_detect_format) {
			BMDDisplayMode orig_mode = mode;
			while(true) {
				mode = select_next_format();
				if (mode==orig_mode) {
					log[log::info] << "Can't set any other format!" << "\n";
					break;
				}
				const char* cm = reinterpret_cast<char*>(&mode);
				log[log::info] << "Trying new format: " << cm[3] << cm[2] << cm[1] << cm[0] << "\n";
				if (restart_streams()) break;
				log[log::info] << "Failed to set the format. trying other one" << "\n";
			}
			manual_detect_timeout=0;
			current_format_name_ = get_mode_name(mode);
		}

	} else /*if (out[0] && out[1])*/ {
		yuri::ubyte_t *data;
		core::pBasicFrame frame;
		yuri::format_t output_format = YURI_FMT_NONE;
		core::pFrameInfo frame_info = yuri::make_shared<core::FrameInfo>();
		frame_info->format = current_format_name_;
		if (pixel_format_map.count(pixel_format)) output_format = pixel_format_map[pixel_format];
		if (videoFrame->GetBytes(reinterpret_cast<void**>(&data))!=S_OK) {
			log[log::error] << "Failed to get data from frame" << "\n";
			return S_OK;
		} else {
			yuri::size_t data_size = videoFrame->GetRowBytes() * height;
//			log[log::info] << "Allocating " << data_size << " bytes for " << height << " lines, " << videoFrame->GetRowBytes() << " bytes each" << "\n";
			frame = allocate_frame_from_memory(data,data_size);
//			if (output_format==YURI_FMT_YUV422) {
//				yuri::ubyte_t *dta = PLANE_RAW_DATA(frame,0);
//				yuri::ubyte_t *dta_end=dta+data_size;
//				while (dta<dta_end) {
//					swap(*dta,*(dta+1));
//					dta+=2;
//				}
//			}
		}
		if (audioPacket && audio_pipe>=0) {
			yuri::size_t samples = audioPacket->GetSampleFrameCount();
			if (samples) {
				yuri::ubyte_t *audio_data;
				if ((res=audioPacket->GetBytes(reinterpret_cast<void**>(&audio_data)))!=S_OK) {
					log[log::error] << "Failed to get data for audio samples! (" << bmerr(res)<<")" << "\n";
				} else {
					core::pBasicFrame audio_frame = allocate_frame_from_memory(audio_data,samples*audio_channels*2);
					push_audio_frame(audio_pipe,audio_frame,YURI_AUDIO_PCM_S16_LE,audio_channels,samples,0,0,0);
				}
			} else {
				log[log::warning] << "Got input frame, but no samples in it" << "\n";
			}
		}


		if (capture_stereo) {
			IDeckLinkVideoFrame3DExtensions *ext;
			IDeckLinkVideoFrame *rightframe;
			if (videoFrame->QueryInterface(IID_IDeckLinkVideoFrame3DExtensions,reinterpret_cast<void**>(&ext))!=S_OK) {
				return S_OK;
			}
			if (ext->GetFrameForRightEye(&rightframe)!=S_OK) {
//				ext->Release();
				return S_OK;
			}
			yuri::ubyte_t *data2;

			if (videoFrame->GetBytes(reinterpret_cast<void**>(&data2))!=S_OK) {
				log[log::error] << "Failed to get data for right eye" << "\n";
				videoFrame->Release();
//				ext->Release();
				return S_OK;
			} else {
				yuri::size_t data_size = rightframe->GetRowBytes() * height;
				core::pBasicFrame frame2 = allocate_frame_from_memory(data2,data_size);
//				yuri::ubyte_t *dta = PLANE_RAW_DATA(frame2,0);
//				yuri::ubyte_t *dta_end=dta+data_size;
//				while (dta<dta_end) {
//					swap(*dta,*(dta+1));
//					dta+=2;
//				}
				frame2->set_info(frame_info);
				if (output_format!=YURI_FMT_NONE) push_video_frame(1,frame2,output_format,width,height,0,1e6*value/scale,0);
				videoFrame->Release();
//				ext->Release();
			}

		}
		frame->set_info(frame_info);
		if (output_format!=YURI_FMT_NONE) push_video_frame(0,frame,output_format,width,height,0,1e6*value/scale,0);
//		push_video_frame(0,frame,YURI_FMT_YUV422,width,height);

	}
	//videoFrame->Release();

	return S_OK;
}


bool DeckLinkInput::init()
{
	if (!init_decklink()) return false;
	IDeckLinkAttributes *attr;
	assert(device);
	device->QueryInterface(IID_IDeckLinkAttributes,reinterpret_cast<void**>(&attr));

	if (device->QueryInterface(IID_IDeckLinkInput,reinterpret_cast<void**>(&input))!=S_OK) {
		log[log::fatal] << "Failed to get input interface" << "\n";
		device->Release();
		device=0;
		return false;
	}
	if (detect_format) {
		bool detection_supported;
		if(attr->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &detection_supported) != S_OK) {
				log[log::error] << "Failed to verify whetehr autodetection is supported" << "\n";
				detect_format = false;
		} else {
			if (!detection_supported) {
				log[log::error] << "Format detection is not supported!" << "\n";
				detect_format = false;
			} else {
				log[log::info] << "Format autodetection is supported" << "\n";
			}
		}
	}
	log[log::debug] << "Initialization OK" << "\n";
	return true;
}

void DeckLinkInput::run()
{
	if (!start_capture()) {
		log[log::fatal] << "Failed to start capture. Bailing out" << "\n";
		exitCode = YURI_EXIT_FINISHED;
		return;
	}
	BasicIOThread::run();
}
bool DeckLinkInput::step()
{
	sleep(latency);
	return true;

}
bool DeckLinkInput::start_capture()
{
	assert(input);
	HRESULT res;
	if (!verify_display_mode()) {
		log[log::warning] << "Failed to verify input mode. Probably unsupported\n";
		return false;
	}
	IDeckLinkConfiguration *cfg;
	if (device->QueryInterface(IID_IDeckLinkConfiguration,reinterpret_cast<void**>(&cfg))!=S_OK) {
		log[log::error]<< "Failed to get cfg handle" << "\n";
	} else {
		if (cfg->SetInt(bmdDeckLinkConfigVideoInputConnection,connection)!=S_OK) {
			log[log::error] << "Failed to set input to SDI" << "\n";
		}
		int64_t x;
		cfg->GetInt(bmdDeckLinkConfigVideoInputConnection,&x);
		log[log::debug] << "Supported connections: " << x << "\n";

	}
	res=S_FALSE;
	if (detect_format) {
		if ((res=input->EnableVideoInput(mode,pixel_format,bmdVideoInputEnableFormatDetection|(capture_stereo?bmdVideoInputDualStream3D:bmdVideoInputFlagDefault)))!=S_OK) {
			log[log::warning] << "Failed to initialize input with format detection enabled. Trying without" << "\n";
			detect_format = false;
		}
	}
	if (res!=S_OK) {
		if ((res=input->EnableVideoInput(mode,pixel_format,capture_stereo?bmdVideoInputDualStream3D:bmdVideoInputFlagDefault))!=S_OK) {
			log[log::error] << "Failed to enable input (" << bmerr(res)<<")" << "\n";
			return false;
		}
	}
	if (audio_enabled) {
		if ((res=input->EnableAudioInput(audio_sample_rate,audio_sample_type,audio_channels))!=S_OK) {
			log[log::error] << "Failed to enable audio (" << bmerr(res)<<")" << "\n";
			return false;
		}
		log[log::info] << "Audio input enabled" << "\n";
	} else {
		input->DisableAudioInput();
	}
	if ((res=input->SetCallback(static_cast<IDeckLinkInputCallback*>(this)))!=S_OK) {
		log[log::error] << "Failed to set callback for input (" << res<<")" << "\n";
		return false;
	}
	if ((res=input->StartStreams())!=S_OK) {
		log[log::error] << "Failed to start streams (" << res<<")" << "\n";
		return false;
	}
	log[log::info] << "Capture started\n";
	return true;
}
bool DeckLinkInput::verify_display_mode()

{
	assert(input);
	mutex::scoped_lock l(schedule_mutex);
	IDeckLinkDisplayMode *dm;
	BMDDisplayModeSupport support;
	BMDVideoInputFlags input_flags = capture_stereo?bmdVideoInputDualStream3D:bmdVideoInputFlagDefault;
	if (input->DoesSupportVideoMode(mode,pixel_format,input_flags,&support,&dm)!=S_OK) return false;
	if (support == bmdDisplayModeNotSupported) return false;
	if (support == bmdDisplayModeSupportedWithConversion) {
		log[log::warning] << "Display mode supported, but conversion is required" << "\n";
	}
	width = dm->GetWidth();
	height = dm->GetHeight();
	dm->GetFrameRate(&value,&scale);
	log[log::info]<<"NTSC "<<(disable_ntsc?"disabled":"enabled")<<", value="<<value<<"\n";
	if (disable_ntsc && value == 1001) return false;
	if (disable_pal && value == 1000) return false;
	const char *modeName;
	if (dm->GetName(&modeName) == S_OK) {
		log[log::info] << "Selected mode " << modeName << ", frame duration "<<value<<"/"<<scale << "\n";
	} else {
		log[log::warning] << "Failed to get mode name!" << "\n";
	}
	if (dm->GetFieldDominance()!=bmdProgressiveFrame) {
		if(disable_interlaced) return false;
		log[log::info] << "Selected format is interlaced, with " << (dm->GetFieldDominance()==bmdLowerFieldFirst?"lower field first":dm->GetFieldDominance()==bmdUpperFieldFirst?"upper field first":"unknows fields") << "\n";
	} else if (disable_progressive) return false;
	return true;
}

BMDDisplayMode DeckLinkInput::select_next_format()
{
	BMDDisplayMode new_mode=0;
	for (std::map<std::string, BMDDisplayMode, yuri::core::compare_insensitive>::iterator i=mode_strings.begin();
			i!=mode_strings.end();++i) {
		if (i->second == mode) {
			if (++i != mode_strings.end()) new_mode=i->second;
			else new_mode = mode_strings.begin()->second;
			break;
		}
	}
	if (!new_mode) new_mode=mode_strings.begin()->second;
	return new_mode;
}

bool DeckLinkInput::restart_streams()
{
	input->StopStreams();
//	input->DisableAudioInput();
//	input->DisableVideoInput();
	return start_capture();
}
bool DeckLinkInput::set_param(const core::Parameter &p)
{
	using boost::iequals;
	if (iequals(p.name, "format_detection")) {
		detect_format=p.get<bool>();
	} else if (iequals(p.name, "force_detection")) {
		manual_detect_format=p.get<yuri::uint_t>();
	} else if (iequals(p.name, "stereo")) {
		capture_stereo=p.get<bool>();
	} else if (iequals(p.name, "disable_ntsc")) {
		disable_ntsc=p.get<bool>();
	} else if (iequals(p.name, "disable_pal")) {
		disable_pal=p.get<bool>();
	} else if (iequals(p.name, "disable_interlaced")) {
		disable_interlaced=p.get<bool>();
	} else if (iequals(p.name, "disable_progressive")) {
		disable_progressive=p.get<bool>();
	} else return DeckLinkBase::set_param(p);

	return true;
}



}

}

