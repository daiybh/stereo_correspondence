/*!
 * @file 		AudioFrame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "AudioFrame.h"

namespace yuri {
namespace core {


AudioFrame::AudioFrame(format_t format, size_t channel_count, size_t sampling_frequency)
:Frame(format), channel_count_(channel_count), sampling_frequency_(sampling_frequency)
{

}

AudioFrame::~AudioFrame() noexcept {}

void AudioFrame::set_channel_count(size_t channel_count)
{
	channel_count_ = channel_count;
}

void AudioFrame::set_sampling_frequency(size_t sampling_frequency)
{
	sampling_frequency_ = sampling_frequency;
}

}
}

