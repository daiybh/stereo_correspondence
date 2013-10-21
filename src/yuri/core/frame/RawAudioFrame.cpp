/*
 * RawAudioFrame.cpp
 *
 *  Created on: 21.10.2013
 *      Author: neneko
 */

#include "RawAudioFrame.h"
#include "raw_audio_frame_params.h"
namespace yuri {
namespace core {

RawAudioFrame::RawAudioFrame(format_t format, size_t channel_count, size_t sampling_frequency)
:AudioFrame(format, channel_count, sampling_frequency)
{
	const auto& fi = raw_audio_format::get_format_info(get_format());
	sample_size = fi.bits_per_sample * channel_count;
}

RawAudioFrame::~RawAudioFrame() noexcept
{

}
pRawAudioFrame RawAudioFrame::create_empty(format_t format, size_t channel_count, size_t sampling_frequency, const uint8_t* data, size_t size)
{
	return create_empty(format, channel_count, sampling_frequency, uvector<uint8_t>(data,data+size));
}
pRawAudioFrame RawAudioFrame::create_empty(format_t format, size_t channel_count, size_t sampling_frequency, uvector<uint8_t>&& data)
{
	pRawAudioFrame frame = make_shared<RawAudioFrame>(format, channel_count, sampling_frequency);
	frame->set_data(std::move(data));
	return frame;
}


void RawAudioFrame::set_data(uvector<uint8_t>&& data)
{
	if (data.size() % (sample_size/8)) throw std::runtime_error("Wring data size!!!!!");
	data_ = std::move(data);
}



// TODO implement
pFrame RawAudioFrame::do_get_copy() const
{
	return {};
}
size_t RawAudioFrame::do_get_size() const noexcept
{
	return size();
}
}
}
