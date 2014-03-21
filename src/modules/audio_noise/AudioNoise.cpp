/*!
 * @file 		AudioNoise.cpp
 * @author 		<Your name>
 * @date		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "AudioNoise.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include "yuri/core/frame/raw_audio_frame_params.h"
#include "yuri/core/frame/RawAudioFrame.h"
namespace yuri {
namespace audio_noise {


IOTHREAD_GENERATOR(AudioNoise)

MODULE_REGISTRATION_BEGIN("audio_noise")
		REGISTER_IOTHREAD("audio_noise",AudioNoise)
MODULE_REGISTRATION_END()

core::Parameters AudioNoise::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("AudioNoise");
	p["format"]["Output format"]="s16_le";
	p["amplitude"]["Amplitude of output signal <0.0, 1.0>"]=1.0;
	p["channels"]["Number of channels"]=1;
	p["frequency"]["Sampling frequency"]=48000;
	return p;
}

struct AudioNoise::frame_generator_t {
	virtual core::pFrame generate(size_t sample_count, std::random_device& rand) = 0;
	virtual ~frame_generator_t() {}
};

namespace {

template<typename T, format_t format>
struct int_noise_generator: AudioNoise::frame_generator_t {
	int_noise_generator(double amplitude, size_t channels, size_t sampling):
		dist(std::numeric_limits<T>::min()*amplitude,std::numeric_limits<T>::max()*amplitude),
		channels(channels),sampling(sampling)
	{}
	virtual core::pFrame generate(size_t sample_count, std::random_device& rand) override
	{
		uvector<uint8_t> data (sample_count*channels*sizeof(T));
		T* data_start = reinterpret_cast<T*>(data.data());
		const T* data_end = data_start +  sample_count*channels;
		while(data_start<data_end) {
			*data_start++=dist(rand);
		}
		auto frame = core::RawAudioFrame::create_empty(format,channels, sampling, std::move(data));
		return frame;
	}
	std::uniform_int_distribution<T> dist;
	size_t channels;
	size_t sampling;
};


struct float_noise_generator: AudioNoise::frame_generator_t {
	float_noise_generator(double amplitude, size_t channels, size_t sampling):
		dist(-amplitude, amplitude),channels(channels),sampling(sampling)
	{}
	virtual core::pFrame generate(size_t sample_count, std::random_device& rand) override
	{
		uvector<uint8_t> data (sample_count*channels*sizeof(float));
		float* data_start = reinterpret_cast<float*>(data.data());
		const float* data_end = data_start +  sample_count*channels;
		while(data_start<data_end) {
			*data_start++=dist(rand);
		}
		auto frame = core::RawAudioFrame::create_empty(core::raw_audio_format::float_32bit,channels, sampling, std::move(data));
		return frame;
	}
	std::uniform_real_distribution<float> dist;
	size_t channels;
	size_t sampling;
};

}

AudioNoise::AudioNoise(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("audio_noise")),format_(core::raw_audio_format::signed_16bit),
amplitude_(1.0),sampling_frequency_(48000),channels_(2)
{
	IOTHREAD_INIT(parameters)
	using namespace core::raw_audio_format;
	switch (format_) {
		case signed_16bit:
			generator_=make_unique<int_noise_generator<int16_t, signed_16bit>>(amplitude_,channels_,sampling_frequency_);
			break;
		case signed_32bit:
			generator_=make_unique<int_noise_generator<int32_t, signed_32bit>>(amplitude_,channels_,sampling_frequency_);
			break;
		case unsigned_8bit:
			generator_=make_unique<int_noise_generator<uint8_t, unsigned_8bit>>(amplitude_,channels_,sampling_frequency_);
			break;
		case unsigned_16bit:
			generator_=make_unique<int_noise_generator<uint16_t, unsigned_16bit>>(amplitude_,channels_,sampling_frequency_);
			break;
		case unsigned_32bit:
			generator_=make_unique<int_noise_generator<uint32_t, unsigned_32bit>>(amplitude_,channels_,sampling_frequency_);
			break;
		case float_32bit:
			generator_=make_unique<float_noise_generator>(amplitude_,channels_,sampling_frequency_);
			break;
	default:
		throw exception::InitializationFailed("Unsupported format");
	}

}

AudioNoise::~AudioNoise() noexcept
{
}

void AudioNoise::run()
{
	timestamp_t next;

	while(still_running()) {
		timestamp_t now;
		if (now > next) {
			next+=100_ms;
			push_frame(0, generator_->generate(sampling_frequency_/10, rand_));
		} else {
			sleep((next-now)/2);
		}
	}
}
bool AudioNoise::set_param(const core::Parameter& param)
{
	if (param.get_name() == "channels") {
		channels_ = param.get<size_t>();
	} else if (param.get_name() == "frequency") {
		sampling_frequency_ = param.get<size_t>();
	} else if (param.get_name() == "amplitude") {
		amplitude_ = param.get<double>();
	} else if (param.get_name() == "format") {
		format_ = core::raw_audio_format::parse_format(param.get<std::string>());
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace audio_noise */
} /* namespace yuri */
