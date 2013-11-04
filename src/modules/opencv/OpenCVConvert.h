/*!
 * @file 		OpenCV.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef OpenCV_H_
#define OpenCV_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"

// Include this only for being able to define convert_format_map here (needed for register.cpp)
#include "yuri/core/frame/raw_frame_types.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace yuri {
namespace opencv {

class OpenCVConvert: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual ~OpenCVConvert() noexcept;
	OpenCVConvert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
private:
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	core::pFrame do_special_single_step(const core::pRawVideoFrame& frame);
	virtual bool set_param(const core::Parameter& param);
	format_t format_;
};

namespace { // TODO: remove this, it's ugly workaround...
typedef std::pair<format_t, format_t> fmt_pair;
std::map<fmt_pair, int > convert_format_map = {
	{{core::raw_format::rgb24,		core::raw_format::rgba32},	static_cast<int>(CV_BGR2BGRA)},
	{{core::raw_format::rgba32,		core::raw_format::rgb24},		static_cast<int>(CV_BGRA2BGR)},
	{{core::raw_format::bayer_rggb,	core::raw_format::rgb24},		static_cast<int>(CV_BayerBG2RGB)},
	{{core::raw_format::bayer_bggr,	core::raw_format::rgb24},		static_cast<int>(CV_BayerRG2RGB)},
	{{core::raw_format::bayer_grbg,	core::raw_format::rgb24},		static_cast<int>(CV_BayerGB2RGB)},
	{{core::raw_format::bayer_gbrg,	core::raw_format::rgb24},		static_cast<int>(CV_BayerGR2RGB)},
};
}

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* OpenCV_H_ */
