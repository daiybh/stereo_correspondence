/*!
 * @file 		AudioLatency.cpp
 * @author 		<Your name>
 * @date		20.05.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "AudioLatency.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include <cmath>
namespace yuri {
namespace audio_latency {


IOTHREAD_GENERATOR(AudioLatency)

MODULE_REGISTRATION_BEGIN("audio_latency")
		REGISTER_IOTHREAD("audio_latency",AudioLatency)
MODULE_REGISTRATION_END()

core::Parameters AudioLatency::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("AudioLatency");
	p["threshold"]["Threshold"] = 0.5;
	p["cooldown"]["cooldown"] = 5120;
	p["max_peak"]["Maximal peak distance"] = 44100*5;
	return p;
}


AudioLatency::AudioLatency(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("audio_latency")),first_peak_(false),
/*first_frame_(true),*/peak_dist_(0),threshold_(16000),cooldown_(512),sample_count_(0),
max_peak_dist_(44100*5),last_peak_(false),coolness_(2,0)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({	core::raw_audio_format::signed_16bit,
							core::raw_audio_format::float_32bit});

}

AudioLatency::~AudioLatency() noexcept
{
}
namespace {
template<typename T>
T get_threshold(double val);

template<>
float get_threshold<float>(double val)
{
	return static_cast<float>(val);
}

template<typename T>
T get_threshold(double val)
{
	return static_cast<T>(std::numeric_limits<T>::max() * val);
}


}

template<typename T>
void AudioLatency::process_latency(const uint8_t* data_ptr, const size_t sample_count, const size_t sampling_frequency, const size_t skip0, const size_t skip1)
{
	const T* data = reinterpret_cast<const T*>(data_ptr);
	T threshold = get_threshold<T>(threshold_);
	for (size_t i = 0; i < sample_count; ++i) {
		if (first_peak_) peak_dist_++;
		if (peak_dist_ > max_peak_dist_) {
			log[log::warning] << "Max peak distance reached, resetting";
			first_peak_ = false;
			peak_dist_ = 0;
		}
		if (std::abs(*data)>threshold && !coolness_[0]) {
			log[log::info] << "Left " << sample_count_+i << ": " << *data;
			coolness_[0]=cooldown_;
			if (!first_peak_) {
				first_peak_ = true;
				last_peak_ = true;
				peak_dist_ = 0;
			}
			else if (last_peak_ == false){
				log[log::info] << "Peak distance " << peak_dist_ << " (" << peak_dist_*1.0e6/sampling_frequency << " us)";
				first_peak_=false;
			}
		}
		if (coolness_[0]) coolness_[0]--;
		data++;
		data+=skip0;
		if (std::abs(*data)>threshold && !coolness_[1]) {
			log[log::info] << "Right " << sample_count_+i << ": " << *data;
			coolness_[1]=cooldown_;
			if (!first_peak_) {
				first_peak_ = true;
				last_peak_ = false;
				peak_dist_ = 0;
			}
			else if (last_peak_ == true){
				log[log::info] << "Peak distance " << peak_dist_ << " (" << peak_dist_*1.0e6/sampling_frequency << " us)";
				first_peak_=false;
			}
		}
		if (coolness_[1]) coolness_[1]--;
		data++;
		data+=skip1;
	}

}

core::pFrame AudioLatency::do_special_single_step(core::pRawAudioFrame frame)
{
	using namespace core::raw_audio_format;
	size_t skip0 = 0;
	size_t skip1 = 0;
	const size_t channels = frame->get_channel_count();
	if (channels < 2) {
		log[log::warning] << "Input format has only " << channels << " channel...";
		return {};
	}
	if (channels > 2) {
		skip1 = channels-2;
	}
	switch (frame->get_format()) {
		case signed_16bit:
			process_latency<int16_t>(frame->data(), frame->get_sample_count(), frame->get_sampling_frequency(), skip0, skip1);
			break;
		case signed_32bit:
			process_latency<int32_t>(frame->data(), frame->get_sample_count(), frame->get_sampling_frequency(), skip0, skip1);
			break;
		case float_32bit:
			process_latency<float>(frame->data(), frame->get_sample_count(), frame->get_sampling_frequency(), skip0, skip1);
			break;
		default:
			log[log::warning] << "Unsupported format";
			break;
	}
	return {};
}
bool AudioLatency::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(threshold_, "threshold")
			(cooldown_, "cooldown")
			(max_peak_dist_, "max_peak"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace audio_latency */
} /* namespace yuri */

