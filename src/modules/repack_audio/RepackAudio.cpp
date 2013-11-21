/*!
 * @file 		RepackAudio.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "RepackAudio.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace repack_audio {

REGISTER("repack_audio",RepackAudio)

IO_THREAD_GENERATOR(RepackAudio)

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::pParameters RepackAudio::configure()
{
	core::pParameters p = core::IOThread::configure();
	p->set_description("Repack audio.");
	//(*p)["fps"]["Framerate of output"]=24.0;
	(*p)["samples"]["Number of samples in output chunk"]=2000;
	(*p)["channels"]["Number of channels in output chunk"]=2;
	p->set_max_pipes(1,1);
	return p;
}


RepackAudio::RepackAudio(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("repack")),samples_(2000,0)
,samples_missing_(2000),total_samples_(2000),channels_(2),current_format_(YURI_AUDIO_PCM_S16_BE)
{
	IO_THREAD_INIT("RepackAudio")
	samples_.resize(total_samples_*channels_*2,0);
}

RepackAudio::~RepackAudio()
{
}
size_t RepackAudio::store_samples(const ubyte_t* start, size_t count)
{
	if (!total_samples_) return count;
	size_t stored = 0;
	while (count > 0) {
		const size_t to_copy = std::min(count, samples_missing_);
		const size_t sample_offset = total_samples_ - samples_missing_;
		std::copy(start,start+2*to_copy*channels_,samples_.begin()+2*sample_offset*channels_);
		count -= to_copy;
		stored += to_copy;
		samples_missing_ -= to_copy;
		if (!samples_missing_) push_current_frame();
	}
	return stored;

}

void RepackAudio::push_current_frame()
{
	core::pBasicFrame frame = allocate_frame_from_memory(&samples_[0],total_samples_*2*channels_);
	frame->set_parameters(current_format_,0,0,channels_,total_samples_);
	push_raw_audio_frame(0,frame);
	samples_missing_ = total_samples_;
}
bool RepackAudio::step()
{
	core::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		current_format_ = frame->get_format();
		if (current_format_!=YURI_AUDIO_PCM_S16_BE &&
				current_format_!=YURI_AUDIO_PCM_S16_LE) {
			log[log::warning] << "Unsupported format. Only 16bit formats supported";
			return true;
		}
		if (frame->get_channel_count() != channels_) {
			log[log::warning] << "Expected " << channels_<<" channels, but got: " << frame->get_channel_count();
			return true;
		}
		const ubyte_t* data = PLANE_RAW_DATA(frame,0);
		size_t available_samples = frame->get_sample_count();
		store_samples(data, available_samples);

	}
	return true;
}
bool RepackAudio::set_param(const core::Parameter& param)
{
	if (param.name == "samples") {
		samples_missing_ = param.get<size_t>();
		total_samples_ = samples_missing_;
	} else if (param.name == "channels") {
		channels_ = param.get<size_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
