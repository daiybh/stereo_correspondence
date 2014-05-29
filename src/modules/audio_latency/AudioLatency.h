/*!
 * @file 		AudioLatency.h
 * @author 		<Your name>
 * @date 		20.05.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef AUDIOLATENCY_H_
#define AUDIOLATENCY_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawAudioFrame.h"

namespace yuri {
namespace audio_latency {

class AudioLatency: public core::SpecializedIOFilter<core::RawAudioFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawAudioFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	AudioLatency(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~AudioLatency() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawAudioFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	template<typename T>
	void process_latency(const uint8_t* data_ptr, const size_t sample_count, const size_t sampling_frequency, const size_t skip0 = 0, const size_t skip1 = 0);
	bool first_peak_;
	size_t peak_dist_;
	double threshold_;
	int_fast32_t cooldown_;
	size_t sample_count_;
	size_t max_peak_dist_;
	bool last_peak_; // 0 - left, 1 - right;
	std::vector<int_fast32_t> coolness_;
};

} /* namespace audio_latency */
} /* namespace yuri */
#endif /* AUDIOLATENCY_H_ */
