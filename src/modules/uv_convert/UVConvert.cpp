/*!
 * @file 		UVConvert.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		31.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVConvert.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/thread/ConverterRegister.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include <unordered_map>
extern "C" {
#include "video_codec.h"
}

namespace yuri {
namespace uv_convert {


namespace {
using decoder_map_t = std::unordered_map<std::pair<format_t, format_t>,  decoder_t>;

//utils::Singleton<decoder_map_t> supported_decoders;

decoder_map_t& get_map()
{
	static mutex m;
	static decoder_map_t supported_decoders;
	lock_t _(m);
	if (supported_decoders.empty()) {
		for (const line_decode_from_to* dec = line_decoders; dec->line_decoder != 0; ++dec) {
			auto p = std::make_pair(ultragrid::uv_to_yuri(dec->from), ultragrid::uv_to_yuri(dec->to));
			if (p.first == 0 || p.second == 0) continue;
			supported_decoders[p]=dec->line_decoder;
		}
	}
	return supported_decoders;
}

size_t get_cost(format_t from_, format_t to_)
{
	codec_t from = ultragrid::yuri_to_uv(from_);
	codec_t to = ultragrid::yuri_to_uv(to_);
	auto cif = codec_info[from];
	auto cit = codec_info[to];
	if (cif.rgb && cit.rgb) return 9; // For RGB conversions
	if (cif.rgb || cit.rgb) return 19; // RGB and some other
	if (cif.bpp == cit.bpp) return 9; // YUYV -> UYVY and similar
	return 14; //
}


}



IOTHREAD_GENERATOR(UVConvert)

MODULE_REGISTRATION_BEGIN("uv_convert")
		REGISTER_IOTHREAD("uv_convert",UVConvert)
		for (const auto&x: get_map()) {
			REGISTER_CONVERTER(x.first.first, x.first.second, "uv_convert", get_cost(x.first.first, x.first.second))
		}
MODULE_REGISTRATION_END()

core::Parameters UVConvert::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("UVConvert");
	p["format"]["Target format"] = "YUV";
	return p;
}

namespace {

core::pFrame convert_raw_frame(const core::pRawVideoFrame& frame_in, format_t target)
{
	if (!frame_in) return {};
	auto p = std::make_pair(frame_in->get_format(), target);
	const auto& fmap = get_map();
	auto it = fmap.find(p);
	if (it == fmap.end()) return {};

	decoder_t decoder = it->second;
	if (!decoder) return {};

	resolution_t res = frame_in->get_resolution();
	core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(target, res);

	const uint8_t * src = PLANE_RAW_DATA(frame_in,0);
	uint8_t * dest = PLANE_RAW_DATA(frame_out,0);

	size_t linesize_in 	= PLANE_DATA(frame_in,0).get_line_size();
	size_t linesize_out = PLANE_DATA(frame_out,0).get_line_size();

	for (size_t line = 0; line < res.height; ++line) {
		decoder(dest, src, linesize_out, 0, 8, 16);
		dest += linesize_out;
		src += linesize_in;
	}
	return frame_out;
}

}

UVConvert::UVConvert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("uv_convert"))
{
	IOTHREAD_INIT(parameters)
	auto m = get_map();
	for (const auto& p: m) {
		log[log::info] << "Found decoder " << ultragrid::yuri_to_uv_string(p.first.first)
		<< " to " << ultragrid::yuri_to_uv_string(p.first.second);
	}
}

UVConvert::~UVConvert() noexcept
{
}


core::pFrame UVConvert::do_special_single_step(const core::pRawVideoFrame& frame)
{
	return convert_raw_frame(frame, format_);
}
core::pFrame UVConvert::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	return convert_raw_frame(dynamic_pointer_cast<core::RawVideoFrame>(input_frame), target_format);
}


bool UVConvert::set_param(const core::Parameter& param)
{
	if (param.get_name()=="format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
	return true;
}

} /* namespace uv_convert */
} /* namespace yuri */
