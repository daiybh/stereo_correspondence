/*
 * AudioFrame.cpp
 *
 *  Created on: 21.10.2013
 *      Author: neneko
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

