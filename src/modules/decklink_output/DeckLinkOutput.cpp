/*
 * DeckLinkOutput.cpp
 *
 *  Created on: Sep 15, 2011
 *      Author: worker
 */

#include "DeckLinkOutput.h"
#include "yuri/core/Module.h"

namespace yuri {

namespace decklink {

REGISTER("decklink_output",DeckLinkOutput)

IO_THREAD_GENERATOR(DeckLinkOutput)

core::pParameters DeckLinkOutput::configure()
{
	core::pParameters p = DeckLinkBase::configure();
	p->set_description("Outputs input video to BlackMagic Design device (Eclipse, Intensity, ...)");
	(*p)["format"]["Output format (1080p25, etc)"]="1080p25";
	(*p)["prebuffer"]["Number of frames to prefill in the output buffer."
	                  "Minimal reasonable value is 2. "
	                  "Setting it to higher values may increase playback fluency, but will increase latency"]=3;
	(*p)["connection"]["Output connection (HDMI, SDI, SVideo, ...). Please note that enabling one output will also enable other compatible outputs"]=std::string("HDMI");
	(*p)["sync"]["Use synchronous frame display."]=true;
	(*p)["stereo"]["Output stereo image."]=false;
	(*p)["format_detection"]["Try to detect video format."]=1;
	return p;
}


DeckLinkOutput::DeckLinkOutput(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		DeckLinkBase(log_,parent,1,0,parameters,"DeckLinkOutput"),output(0),
		output_connection(bmdVideoConnectionHDMI),
		/*act_oframe(0),back_oframe(0),*/frame_number(0),enabled(false),
		buffer_num(5),prebuffer(5),sync(true),stereo_mode(false),stereo_usable(false)
{
	IO_THREAD_INIT("Decklink Output")
	if (stereo_mode) resize(2,0);
	if (audio_enabled) resize(3,0);
	if (!init()) {
		throw exception::InitializationFailed("Failed to initialize DeckLink device");
	}
}

DeckLinkOutput::~DeckLinkOutput()
{

}

bool DeckLinkOutput::init()
{
	if (!init_decklink()) return false;
	IDeckLinkAttributes *attr;
	device->QueryInterface(IID_IDeckLinkAttributes,reinterpret_cast<void**>(&attr));
	//int64_t video_con;
	//attr->GetInt(BMDDeckLinkVideoOutputConnections,&video_con);
	if (device->QueryInterface(IID_IDeckLinkOutput,reinterpret_cast<void**>(&output))!=S_OK) {
		log[log::fatal] << "Failed to get output interface\n";
		device->Release();
		device=0;
		return false;
	}

	if (prebuffer >= buffer_num) buffer_num = prebuffer+1;

	return true;
}
void DeckLinkOutput::run()
{
	log[log::debug] << "Starting up\n";
	if (!verify_display_mode()) {
		log[log::error] << "Failed to verify display mode\n";
		return;
	}
	if (!start_stream()) {
		log[log::error] << "Failed to start stream\n";
		return;
	}
	BasicIOThread::run();
	stop_stream();
}


bool DeckLinkOutput::set_param(const core::Parameter &p)
{
	using boost::iequals;
	if (iequals(p.name, "format_detection")) {
		detect_format=p.get<bool>();
	} else if (iequals(p.name, "prebuffer")) {
		prebuffer=p.get<yuri::ushort_t>();
	} else if (iequals(p.name, "sync")) {
		sync=p.get<bool>();
	} else if (iequals(p.name, "stereo")) {
		stereo_mode=p.get<bool>();
	} return DeckLinkBase::set_param(p);
	return true;
}




bool DeckLinkOutput::verify_display_mode()
{
	assert(output);
	mutex::scoped_lock l(schedule_mutex);
	IDeckLinkDisplayMode *dm;
	BMDDisplayModeSupport support;
	stereo_usable= false;
	if (stereo_mode) {
		if (output->DoesSupportVideoMode(mode,pixel_format,bmdVideoOutputDualStream3D,&support,&dm)!=S_OK) {
			log[log::warning] << "Selected format does not work with 3D. Re-trying without 3D support\n";
			if (output->DoesSupportVideoMode(mode,pixel_format,bmdVideoOutputFlagDefault,&support,&dm)!=S_OK) return false;
		} else {
			stereo_usable = true;
		}

	} else if (output->DoesSupportVideoMode(mode,pixel_format,bmdVideoOutputFlagDefault,&support,&dm)!=S_OK) return false;
	if (support==bmdDisplayModeNotSupported) return false;
	if (support == bmdDisplayModeSupportedWithConversion) {
		log[log::warning] << "Display mode supported, but conversion is required\n";
	}
	width = dm->GetWidth();
	height = dm->GetHeight();
	dm->GetFrameRate(&value,&scale);
	const char       *modeName;
	if (dm->GetName(&modeName) == S_OK) {
		log[log::info] << "Selected mode " << modeName << "\n";
	} else {
		log[log::warning] << "Failed to get mode name!\n";
	}
	//if (act_oframe) act_oframe->Release();
	//if (back_oframe) back_oframe->Release();
	yuri::uint_t linesize;
	switch (pixel_format) {
		case bmdFormat8BitYUV: linesize=width*2;break;
		case bmdFormat8BitARGB: linesize=width*4;break;
		case bmdFormat10BitYUV: linesize=16*(width/6+(width%6?1:0));break;
		default: log[log::error] << "Unsupported pixel format\n";return false;
	}
	out_frames.clear();
	//shared_ptr<IDeckLinkMutableVideoFrame> f;
	for (yuri::size_t i=0;i<buffer_num;++i) {
		shared_ptr<DeckLink3DVideoFrame>  f(new DeckLink3DVideoFrame(width,height,pixel_format,bmdFrameFlagDefault));
		if (stereo_mode) {
			shared_ptr<DeckLink3DVideoFrame> f2(new DeckLink3DVideoFrame(width,height,pixel_format,bmdFrameFlagDefault));
			//f->set_packing_format(bmdVideo3DPackingFramePacking);
			f->set_packing_format(bmdVideo3DPackingLeftOnly);
			//f->set_packing_format(bmdVideo3DPackingTopAndBottom);
			f->add_right(f2);
		}
		/*if (output->CreateVideoFrame(width,height,linesize,pixel_format,bmdFrameFlagDefault,&f)!=S_OK) {
			log[log::error] << "Failed to allocate video frame" << endl;
			return false;
		}
		shared_ptr<DeckLink3DVideoFrame> pf(f,decklink_deleter);*/
		out_frames.push_back(f);
	}
	return true;
}

void DeckLinkOutput::schedule(bool force)
{
	mutex::scoped_lock l(schedule_mutex);
	//if (!act_oframe) {
	if (!out_frames.size()) {
		log[log::error] << "No frame to schedule!!\n";
		return;

	}
	HRESULT res;
	bool act;
	output->IsScheduledPlaybackRunning(&act);
	if (!act && !force) return;
	//if ((res=output->ScheduleVideoFrame(act_oframe,value*frame_number,value,scale))!=S_OK) {
	if ((res=output->ScheduleVideoFrame(do_get_active_buffer().get(),value*frame_number,value,scale))!=S_OK) {
		log[log::error] << "Failed to schedule frame "<<frame_number << "! " << bmerr(res) << " (" << HRESULT_CODE(res) << ")\n";
	}
	log[log::debug] << "Scheduled frame " << frame_number << "\n";
	frame_number++;
}
HRESULT STDMETHODCALLTYPE	DeckLinkOutput::ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
{
	schedule(false);
	return S_OK;
}

HRESULT STDMETHODCALLTYPE	DeckLinkOutput::ScheduledPlaybackHasStopped ()
{
	enabled = false;
	return S_OK;
}

bool DeckLinkOutput::start_stream()
{
	assert(output);
	HRESULT res;
	BMDVideoOutputFlags flags = bmdVideoOutputFlagDefault;
	if (stereo_usable) flags|=bmdVideoOutputDualStream3D;

	if ((res=output->EnableVideoOutput(mode,flags))!=S_OK) {
		log[log::error] << "Failed to enable display mode! Error: "<< bmerr(res) << " ("<<HRESULT_CODE(res)<<")\n";
		return false;
	}
	if (audio_enabled) {
		output->EnableAudioOutput(bmdAudioSampleRate48kHz,bmdAudioSampleType16bitInteger,2,bmdAudioOutputStreamContinuous);
	} else {
		output->DisableAudioOutput();
	}
	return true;
}
bool DeckLinkOutput::stop_stream()
{
	assert(output);
	HRESULT res;
	output->DisableAudioOutput();
	output->DisableVideoOutput();
	BMDTimeValue t;
	if (!sync) {
		output->StopScheduledPlayback(0,&t,scale);
		bool r;
		output->IsScheduledPlaybackRunning(&r);
		if (r) {
			log[log::warning] << "Playback still running after being stopped\n";
		}
	}
	/*if ((res=output->EnableVideoOutput(mode,bmdVideoOutputFlagDefault))!=S_OK) {
		log[log::error] << "Failed to enable display mode! Error: "<< bmerr(res) << " ("<<HRESULT_CODE(res)<<")" << endl;
		return false;
	}*/
	return true;
}
bool DeckLinkOutput::enable_stream()
{
	if (enabled) return true;
	for (int i=0;i<prebuffer;++i) schedule(true);
	output->SetScheduledFrameCompletionCallback(this);
	output->StartScheduledPlayback(0,scale,1.0);

	enabled=true;
	return true;
}

bool DeckLinkOutput::step()
{
	core::pBasicFrame f;
	if (!in[0]) return true;
	/*bool new_frame = false;
	while (f=in[0]->pop_frame()) {
		frame = f;
		new_frame=true;
		break;
	}
	if (!new_frame) return true;*/
	if (!frame) frame = in[0]->pop_frame();
	if (stereo_usable) {
		if (!in[1]) return true;
		if (!frame2) frame2 = in[1]->pop_frame();
	}
	if (!frame || (stereo_usable && !frame2)) return true;
	BMDPixelFormat pfmt = 0;
	switch (frame->get_format()) {
		case YURI_FMT_UYVY422:pfmt = bmdFormat8BitYUV;break;
		case YURI_FMT_V210:pfmt = bmdFormat10BitYUV;break;
		default: pfmt = 0;break;
	}
	if (!pfmt) {
		log[log::error] << "Unsupported pixel format: "<<core::BasicPipe::get_format_string(frame->get_format());
		frame.reset();
		frame2.reset();
		return true;
	}
	if (stereo_usable) {
		if (frame->get_format() != frame2->get_format()) {
			log[log::error] << "Frames for left and right eye has different pixel formats\n";
			frame.reset();
			frame2.reset();
			return true;
		}
	}
	core::pFrameInfo fi = frame->get_info();
	if (detect_format && fi) {
		BMDDisplayMode m = parse_format(fi->format);
		if (m==bmdModeUnknown) {
			log[log::warning] << "Format specified in incoming frame is not supported!";
		} else if (m!=mode) {
			pixel_format = pfmt;
			mode = m;
			if (!verify_display_mode()) {
				log[log::error] << "Failed to verify display mode for incoming frame\n";
				frame.reset();
				frame2.reset();
				return true;
			}
			log[log::info] << "Format changed to " << fi->format;
		}
	}
	if (pfmt!=pixel_format) {
		pixel_format=pfmt;
		if (!verify_display_mode()) {
			log[log::error] << "Failed to verify display mode\n";
			frame.reset();
			frame2.reset();
			return true;
		}
	}
	shared_ptr<DeckLink3DVideoFrame> pf = get_next_buffer();
	fill_frame(frame,pf);
	if (stereo_usable) fill_frame(frame2,pf->get_right());
	if (audio_enabled && in[2]) {
		core::pBasicFrame audio_frame=in[2]->pop_frame();
		if (audio_frame) {
			uint32_t writen;
			std::vector<yuri::sshort_t> samples(audio_frame->get_sample_count()*2);
			if (audio_frame->get_format() == YURI_AUDIO_PCM_S24_BE) {
//				log[log::warning] << "Got S24BE"  << endl;
				const yuri::ubyte_t *data = PLANE_RAW_DATA(audio_frame,0);
				yuri::size_t zeroes=0;
				yuri::size_t chan = audio_frame->get_channel_count();
				yuri::size_t skip =3*chan;
				yuri::sint_t sample;
				for (int i=0;i<audio_frame->get_sample_count();++i) {
					sample = (data[0]<<16)|(data[1]<<8)|data[2];
					samples[2*i]=sample>>8;
					//samples[4*i+1]=data[1];//>>8)|(data[0]);
					//if (samples[2*i]==0) zeroes++;
					if (chan>1) samples[2*i+1] << (data[3]<<8)|(data[4]);
					else samples[2*i+1] = 0;
					//samples[2*i+1]=((0xFF*i)%480);
//					samples[4*i+3]=((0xFF*i)%480)>>8;
					data+=skip;
				}
//				log[log::info] << "Zeroes " << zeroes << "/" <<audio_frame->get_sample_count()<<endl;
			} else {
				log[log::warning] << "Unsupported input format: " << core::BasicPipe::get_format_string(audio_frame->get_format()) << "\n";
			}
			if (samples.size()) {
				output->WriteAudioSamplesSync(&samples[0],audio_frame->get_sample_count(),&writen);
			}
		} else {
			log[log::warning] << "No audio\n";
		}
	}
	if (sync) {
		output->DisplayVideoFrameSync(pf.get());
	} else {
		rotate_buffers();
		enable_stream();
	}
	frame.reset();
	frame2.reset();

	return true;
}
shared_ptr<DeckLink3DVideoFrame> DeckLinkOutput::get_active_buffer()
{
	mutex::scoped_lock l(schedule_mutex);
	return do_get_active_buffer();
}
shared_ptr<DeckLink3DVideoFrame> DeckLinkOutput::do_get_active_buffer()
{

	assert (out_frames.size());
	return out_frames[0];
}
shared_ptr<DeckLink3DVideoFrame> DeckLinkOutput::get_next_buffer()
{
	mutex::scoped_lock l(schedule_mutex);
	return do_get_next_buffer();
}
shared_ptr<DeckLink3DVideoFrame> DeckLinkOutput::do_get_next_buffer()
{
	assert (out_frames.size()>1);
	return out_frames[1];
}
void DeckLinkOutput::rotate_buffers()
{
	mutex::scoped_lock l(schedule_mutex);
	do_rotate_buffers();
}
void DeckLinkOutput::do_rotate_buffers()
{
	shared_ptr<DeckLink3DVideoFrame> pf = out_frames[0];
	out_frames.pop_front();
	out_frames.push_back(pf);
}
bool DeckLinkOutput::fill_frame(core::pBasicFrame source, shared_ptr<DeckLink3DVideoFrame> output)
{
	assert(source);
	yuri::uint_t line_width = 0, copy_width, target_width,copy_lines;
	FormatInfo_t fi = core::BasicPipe::get_format_info(source->get_format());
	if (fi) {
		line_width=source->get_width()*(fi->bpp >> 3);
	}

	target_width=width*(fi->bpp >> 3);	
	if (source->get_format() == YURI_FMT_V210) {
		const size_t w = source->get_width();
		yuri::size_t in_line_width6 = w/6;// + (w%6?1:0);
		if (w==1280) in_line_width6 = 216;
		yuri::size_t line_width6 = width/6 + (width%6?1:0);
		if (width==1280) line_width6 = 216;
		line_width=in_line_width6*16;
		target_width=line_width6*16;
	}
	if (line_width == 0) {
		return false;
	}

	copy_width=std::min(line_width,target_width);
	copy_lines=std::min(source->get_height(),static_cast<size_t>(height));
	log[log::debug] << "Copying " << copy_lines << " lines of " << copy_width << " bytes (of " << line_width << " total bytes)\n";
	if (line_width*copy_lines>PLANE_SIZE(source,0)) {
		log[log::warning] << "not enough data to copy!!!";
		return false;
	}
	yuri::ubyte_t *data = PLANE_RAW_DATA(source,0);
	yuri::ubyte_t *data2;

	//back_oframe->GetBytes(reinterpret_cast<void**>(&data2));
	output->GetBytes(reinterpret_cast<void**>(&data2));
	assert(data2);
	for (int h=0;h<copy_lines;++h) {
		memcpy(data2,data,copy_width);
		data+=line_width;
		data2+=target_width;
	}
	return true;
}
}
}
