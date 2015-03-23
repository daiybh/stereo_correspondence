/*!
 * @file 		AudioVisualization.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "AudioVisualization.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/frame/raw_audio_frame_types.h"

namespace yuri {
namespace audio_visualization {


IOTHREAD_GENERATOR(AudioVisualization)

MODULE_REGISTRATION_BEGIN("audio_visualization")
		REGISTER_IOTHREAD("audio_visualization",AudioVisualization)
MODULE_REGISTRATION_END()

core::Parameters AudioVisualization::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("AudioVisualization");
	p["zoom"]["Zoom in time dimension"]=1;
	p["height"]["Height of visualized waveform. Set to 0 to use full height"]=0;
	return p;
}


AudioVisualization::AudioVisualization(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_, parent, 1, std::string("audio_visualization")),
height_(0),zoom_(1)
{
	IOTHREAD_INIT(parameters)
}

AudioVisualization::~AudioVisualization() noexcept
{
}

void AudioVisualization::run()
{
	video_converter_ = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
	base_type::run();
}

namespace {
void set_val(uint8_t*)
{
}


template<uint8_t v, uint8_t... vals>
void set_val(uint8_t* dv)
{
	*dv=v;
	set_val(dv+1);
}

template<typename sample_t, uint8_t... vals>
void process_frame(core::pRawVideoFrame& video_frame, const core::pRawAudioFrame& audio_frame, size_t h, size_t zoom)
{
	constexpr const auto size = sizeof...(vals);
	const auto res = video_frame->get_resolution();
	const auto chan = audio_frame->get_channel_count();
	auto d = reinterpret_cast<sample_t*>(audio_frame->data());
	const auto d_end = d + chan * audio_frame->get_sample_count();
	auto dv = PLANE_RAW_DATA(video_frame, 0);
	dimension_t x = 0;
	while ((d < d_end) && (x < res.width)) {
		auto val = clip_value<dimension_t, dimension_t>(
				static_cast<position_t>(static_cast<double>(h) * *d / std::numeric_limits<sample_t>::max() / 2) + (h / 2),
				0, res.height-1);
		set_val<vals...>(dv + (val*size*res.width));
		dv+=size;
		++x;
		d+=chan*zoom;
	}
}


}
std::vector<core::pFrame> AudioVisualization::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawAudioFrame> frames)
{
	auto fv = std::dynamic_pointer_cast<core::RawVideoFrame>(
			video_converter_->convert_to_cheapest(std::get<0>(frames),
					{core::raw_format::rgb24, core::raw_format::bgr24,

							core::raw_format::rgba32, core::raw_format::bgra32,
							core::raw_format::argb32, core::raw_format::abgr32,

							core::raw_format::yuv444,
							core::raw_format::y8}));
	if (!fv) return {};
	auto video_frame = get_frame_unique(fv);
	auto audio_frame = std::get<1>(frames);

	if (audio_frame->get_format() != core::raw_audio_format::signed_16bit) {
		log[log::warning] << "Unsupported audio format";
		return {};
	}

	const auto res = video_frame->get_resolution();
	const auto h = height_>0?height_:res.height;

	switch(video_frame->get_format()) {
		case core::raw_format::rgba32:
		case core::raw_format::bgra32:
		case core::raw_format::argb32:
		case core::raw_format::abgr32:
			process_frame<int16_t, 255, 255, 255, 255>(video_frame, audio_frame, h, zoom_);
					break;
		case core::raw_format::rgb24:
		case core::raw_format::bgr24:
			process_frame<int16_t, 255, 255, 255>(video_frame, audio_frame, h, zoom_);
			break;
		case core::raw_format::yuv444:
			process_frame<int16_t, 255, 127, 127>(video_frame, audio_frame, h, zoom_);
			break;
		case core::raw_format::y8:
			process_frame<int16_t, 255>(video_frame, audio_frame, h, zoom_);
			break;
		default:break;
	}
	return {video_frame};
}

bool AudioVisualization::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(zoom_, 	"zoom")
			(height_, 	"height"))
		return true;
	return base_type::set_param(param);
}

} /* namespace audio_visualization */
} /* namespace yuri */
