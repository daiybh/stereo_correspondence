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
	p["threshold"]["Threshold"] = 16000;
	p["cooldown"]["cooldown"] = 512;
	p["max_peak"]["Maximal peak distance"] = 44100*5;
	return p;
}


AudioLatency::AudioLatency(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("audio_latency")),first_peak_(false),
/*first_frame_(true),*/peak_dist_(0),threshold_(16000),cooldown_(512),sample_count_(0),
max_peak_dist_(44100*5),last_peak_(false),coolness_(2,0)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats({core::raw_audio_format::signed_16bit});

}

AudioLatency::~AudioLatency() noexcept
{
}

core::pFrame AudioLatency::do_special_single_step(const core::pRawAudioFrame& frame)
{
	const int16_t *data = reinterpret_cast<const int16_t*>(frame->data());


	for (size_t i = 0; i < frame->get_sample_count(); ++i) {
		if (first_peak_) peak_dist_++;
		if (peak_dist_ > max_peak_dist_) {
			log[log::warning] << "Max peak distance reached, resetting";
			first_peak_ = false;
			peak_dist_ = 0;
		}
		if (std::abs(*data)>threshold_ && !coolness_[0]) {
			log[log::info] << "Left " << sample_count_+i << ": " << *data;
			coolness_[0]=cooldown_;
			if (!first_peak_) {
				first_peak_ = true;
				last_peak_ = true;
				peak_dist_ = 0;
			}
			else if (last_peak_ == false){
				log[log::info] << "Peak distance " << peak_dist_ << " (" << peak_dist_*1.0e6/44100 << " us)";
				first_peak_=false;
			}
		}
		if (coolness_[0]) coolness_[0]--;
		data++;
		if (std::abs(*data)>threshold_ && !coolness_[1]) {
			log[log::info] << "Right " << sample_count_+i << ": " << *data;
			coolness_[1]=cooldown_;
			if (!first_peak_) {
				first_peak_ = true;
				last_peak_ = false;
				peak_dist_ = 0;
			}
			else if (last_peak_ == true){
				log[log::info] << "Peak distance " << peak_dist_ << " (" << peak_dist_*1.0e6/44100 << " us)";
				first_peak_=false;
			}
		}
		if (coolness_[1]) coolness_[1]--;
		data++;
	}



	return {};
}
bool AudioLatency::set_param(const core::Parameter& param)
{
	if (param.get_name() =="threshold") {
		threshold_ = param.get<int_fast32_t>();
	} else if (param.get_name() == "cooldown") {
		cooldown_ = param.get<int_fast32_t>();
	} else if (param.get_name() == "max_peak") {
		max_peak_dist_ = param.get<size_t>();
	}
	return core::IOThread::set_param(param);
}

} /* namespace audio_latency */
} /* namespace yuri */
