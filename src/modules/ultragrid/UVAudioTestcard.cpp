/*!
 * @file 		UVAudioTestcard.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		22.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVAudioTestcard.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include "yuri/core/utils/array_range.h"
extern "C" {
#include "audio/audio.h"
#include "host.h"
#include "audio/capture/testcard.h"
}
namespace yuri {
namespace uv_audio_testcard {


IOTHREAD_GENERATOR(UVAudioTestcard)

core::Parameters UVAudioTestcard::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVAudioTestcard");
	return p;
}


UVAudioTestcard::UVAudioTestcard(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv_audio_testcard")),device_(nullptr)
{
	IOTHREAD_INIT(parameters)
	set_latency(2_ms);
	//audio_capture_channels = capture_channels_;
	// TODO Ewwww!! Let's fix upstream to take const char*!!
	device_ = audio_cap_testcard_init("");
	if (!device_) throw exception::InitializationFailed("Failed to initialize testcard device");
}

UVAudioTestcard::~UVAudioTestcard() noexcept
{
}

void UVAudioTestcard::run()
{
	audio_frame* frame = nullptr;
	while(still_running()) {
		frame = audio_cap_testcard_read(device_);
		if (!frame || (!frame->data_len)) {
			sleep(get_latency());
			continue;
		}
//		log[log::info] << "Pushing sample with " << frame->bps << " bytes per sample, "
//					<< frame->sample_rate << " samples per second and " << frame->ch_count
//					<< " channels. Data length: " << frame->data_len;
		core::pRawAudioFrame out_frame = core::RawAudioFrame::create_empty(core::raw_audio_format::signed_16bit,
				frame->ch_count,
				frame->sample_rate,
				frame->data,
				frame->data_len);


		size_t zeros = 0;
		for (const auto& x: array_range<char>(frame->data, frame->data_len)) {
			if (!x) zeros++;
		}
		log[log::info] << "There was " << (100.0*static_cast<double>(zeros)/frame->data_len) << "% zeros";
		push_frame(0,out_frame);
	}

	audio_cap_testcard_finish(device_);
}
bool UVAudioTestcard::set_param(const core::Parameter& param)
{
	if (param.get_name() == "channels") {
		capture_channels_ = param.get<size_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace uv_audio_testcard */
} /* namespace yuri */
