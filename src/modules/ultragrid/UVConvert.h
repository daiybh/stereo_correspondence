/*!
 * @file 		UVConvert.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		31.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVCONVERT_H_
#define UVCONVERT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
#include <unordered_map>
extern "C" {
// Included because of decoder_t, needed by decoder_map_t that in turn is now needed by register.cpp
#include "video_codec.h"
}

namespace yuri {
namespace uv_convert {

using decoder_map_t = std::unordered_map<std::pair<format_t, format_t>,  decoder_t>;

class UVConvert: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVConvert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVConvert() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;

	format_t format_;
};


size_t get_cost(format_t from_, format_t to_);
decoder_map_t& get_map();

} /* namespace uv_convert */
} /* namespace yuri */
#endif /* UVCONVERT_H_ */
