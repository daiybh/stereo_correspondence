/*!
 * @file 		OpenCV.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
#include "opencv2/imgproc/types_c.h"

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
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param);
	format_t format_;
};

namespace { // TODO: remove this, it's ugly workaround...
typedef std::pair<format_t, format_t> fmt_pair;
std::map<fmt_pair, int > convert_format_map = {
	{{core::raw_format::rgb24,		core::raw_format::rgba32},		static_cast<int>(CV_RGB2RGBA)},
	{{core::raw_format::rgba32,		core::raw_format::rgb24},		static_cast<int>(CV_RGBA2RGB)},
	{{core::raw_format::bgr24,		core::raw_format::bgra32},		static_cast<int>(CV_BGR2BGRA)},
	{{core::raw_format::bgra32,		core::raw_format::bgr24},		static_cast<int>(CV_BGRA2BGR)},

	{{core::raw_format::bgr24,		core::raw_format::rgba32},		static_cast<int>(CV_BGR2RGBA)},
	{{core::raw_format::rgb24,		core::raw_format::bgra32},		static_cast<int>(CV_RGB2BGRA)},
	{{core::raw_format::rgba32,		core::raw_format::bgr24},		static_cast<int>(CV_RGBA2BGR)},
	{{core::raw_format::bgra32,		core::raw_format::rgb24},		static_cast<int>(CV_BGRA2RGB)},

	{{core::raw_format::rgb24,		core::raw_format::bgr24},		static_cast<int>(CV_RGB2BGR)},
	{{core::raw_format::bgr24,		core::raw_format::rgb24},		static_cast<int>(CV_BGR2RGB)},

	{{core::raw_format::rgb24,		core::raw_format::y8},			static_cast<int>(CV_RGB2GRAY)},
	{{core::raw_format::rgba32,		core::raw_format::y8},			static_cast<int>(CV_RGBA2GRAY)},
	{{core::raw_format::bgr24,		core::raw_format::y8},			static_cast<int>(CV_BGR2GRAY)},
	{{core::raw_format::bgra32,		core::raw_format::y8},			static_cast<int>(CV_BGRA2GRAY)},



	// 8 - 16 bit
	{{core::raw_format::rgb24,		core::raw_format::rgba64},		static_cast<int>(CV_RGB2RGBA)},
	{{core::raw_format::rgba32,		core::raw_format::rgb48},		static_cast<int>(CV_RGBA2RGB)},
	{{core::raw_format::bgr24,		core::raw_format::bgra64},		static_cast<int>(CV_BGR2BGRA)},
	{{core::raw_format::bgra32,		core::raw_format::bgr48},		static_cast<int>(CV_BGRA2BGR)},

	{{core::raw_format::bgr24,		core::raw_format::rgba64},		static_cast<int>(CV_BGR2RGBA)},
	{{core::raw_format::rgb24,		core::raw_format::bgra64},		static_cast<int>(CV_RGB2BGRA)},
	{{core::raw_format::rgba32,		core::raw_format::bgr48},		static_cast<int>(CV_RGBA2BGR)},
	{{core::raw_format::bgra32,		core::raw_format::rgb48},		static_cast<int>(CV_BGRA2RGB)},

	{{core::raw_format::rgb24,		core::raw_format::bgr48},		static_cast<int>(CV_RGB2BGR)},
	{{core::raw_format::bgr24,		core::raw_format::rgb48},		static_cast<int>(CV_BGR2RGB)},

	{{core::raw_format::rgb24,		core::raw_format::y16},			static_cast<int>(CV_RGB2GRAY)},
	{{core::raw_format::rgba32,		core::raw_format::y16},			static_cast<int>(CV_RGBA2GRAY)},
	{{core::raw_format::bgr24,		core::raw_format::y16},			static_cast<int>(CV_BGR2GRAY)},
	{{core::raw_format::bgra32,		core::raw_format::y16},			static_cast<int>(CV_BGRA2GRAY)},

	// 16 - 8 bit
	{{core::raw_format::rgb48,		core::raw_format::rgba32},		static_cast<int>(CV_RGB2RGBA)},
	{{core::raw_format::rgba64,		core::raw_format::rgb24},		static_cast<int>(CV_RGBA2RGB)},
	{{core::raw_format::bgr48,		core::raw_format::bgra32},		static_cast<int>(CV_BGR2BGRA)},
	{{core::raw_format::bgra64,		core::raw_format::bgr24},		static_cast<int>(CV_BGRA2BGR)},

	{{core::raw_format::bgr48,		core::raw_format::rgba32},		static_cast<int>(CV_BGR2RGBA)},
	{{core::raw_format::rgb48,		core::raw_format::bgra32},		static_cast<int>(CV_RGB2BGRA)},
	{{core::raw_format::rgba64,		core::raw_format::bgr24},		static_cast<int>(CV_RGBA2BGR)},
	{{core::raw_format::bgra64,		core::raw_format::rgb24},		static_cast<int>(CV_BGRA2RGB)},

	{{core::raw_format::rgb48,		core::raw_format::bgr24},		static_cast<int>(CV_RGB2BGR)},
	{{core::raw_format::bgr48,		core::raw_format::rgb24},		static_cast<int>(CV_BGR2RGB)},

	{{core::raw_format::rgb48,		core::raw_format::y8},			static_cast<int>(CV_RGB2GRAY)},
	{{core::raw_format::rgba64,		core::raw_format::y8},			static_cast<int>(CV_RGBA2GRAY)},
	{{core::raw_format::bgr48,		core::raw_format::y8},			static_cast<int>(CV_BGR2GRAY)},
	{{core::raw_format::bgra64,		core::raw_format::y8},			static_cast<int>(CV_BGRA2GRAY)},

	// 16- 16 bit
	{{core::raw_format::rgb48,		core::raw_format::rgba64},		static_cast<int>(CV_RGB2RGBA)},
	{{core::raw_format::rgba64,		core::raw_format::rgb48},		static_cast<int>(CV_RGBA2RGB)},
	{{core::raw_format::bgr48,		core::raw_format::bgra64},		static_cast<int>(CV_BGR2BGRA)},
	{{core::raw_format::bgra64,		core::raw_format::bgr48},		static_cast<int>(CV_BGRA2BGR)},

	{{core::raw_format::bgr48,		core::raw_format::rgba64},		static_cast<int>(CV_BGR2RGBA)},
	{{core::raw_format::rgb48,		core::raw_format::bgra64},		static_cast<int>(CV_RGB2BGRA)},
	{{core::raw_format::rgba64,		core::raw_format::bgr48},		static_cast<int>(CV_RGBA2BGR)},
	{{core::raw_format::bgra64,		core::raw_format::rgb48},		static_cast<int>(CV_BGRA2RGB)},

	{{core::raw_format::rgb48,		core::raw_format::bgr48},		static_cast<int>(CV_RGB2BGR)},
	{{core::raw_format::bgr48,		core::raw_format::rgb48},		static_cast<int>(CV_BGR2RGB)},

	{{core::raw_format::rgb48,		core::raw_format::y16},			static_cast<int>(CV_RGB2GRAY)},
	{{core::raw_format::rgba64,		core::raw_format::y16},			static_cast<int>(CV_RGBA2GRAY)},
	{{core::raw_format::bgr48,		core::raw_format::y16},			static_cast<int>(CV_BGR2GRAY)},
	{{core::raw_format::bgra64,		core::raw_format::y16},			static_cast<int>(CV_BGRA2GRAY)},





	/* How is this supposed to be represented in cm::Mat?
	{{core::raw_format::rgb24,		core::raw_format::rgb16},		static_cast<int>(CV_BGR2BGR565)},
	{{core::raw_format::bgr24,		core::raw_format::rgb16},		static_cast<int>(CV_RGB2BGR565)},
	{{core::raw_format::rgb24,		core::raw_format::bgr16},		static_cast<int>(CV_RGB2BGR565)},
	{{core::raw_format::bgr24,		core::raw_format::bgr16},		static_cast<int>(CV_BGR2BGR565)},

	{{core::raw_format::bgr16,		core::raw_format::bgr24},		static_cast<int>(CV_BGR5652BGR)},
	{{core::raw_format::rgb16,		core::raw_format::rgb24},		static_cast<int>(CV_BGR5652BGR)},
	{{core::raw_format::bgr16,		core::raw_format::rgb24},		static_cast<int>(CV_BGR5652RGB)},
	{{core::raw_format::rgb16,		core::raw_format::bgr24},		static_cast<int>(CV_BGR5652RGB)},

	{{core::raw_format::rgba32,		core::raw_format::rgb16},		static_cast<int>(CV_BGRA2BGR565)},
	{{core::raw_format::bgra32,		core::raw_format::rgb16},		static_cast<int>(CV_RGBA2BGR565)},
	{{core::raw_format::rgba32,		core::raw_format::bgr16},		static_cast<int>(CV_RGBA2BGR565)},
	{{core::raw_format::bgra32,		core::raw_format::bgr16},		static_cast<int>(CV_BGRA2BGR565)},

	{{core::raw_format::bgr16,		core::raw_format::bgra32},		static_cast<int>(CV_BGR5652BGRA)},
	{{core::raw_format::rgb16,		core::raw_format::rgba32},		static_cast<int>(CV_BGR5652BGRA)},
	{{core::raw_format::bgr16,		core::raw_format::rgba32},		static_cast<int>(CV_BGR5652RGBA)},
	{{core::raw_format::rgb16,		core::raw_format::bgra32},		static_cast<int>(CV_BGR5652RGBA)},

 */

	{{core::raw_format::bayer_rggb,	core::raw_format::rgb24},		static_cast<int>(CV_BayerBG2RGB)},
	{{core::raw_format::bayer_bggr,	core::raw_format::rgb24},		static_cast<int>(CV_BayerRG2RGB)},
	{{core::raw_format::bayer_grbg,	core::raw_format::rgb24},		static_cast<int>(CV_BayerGB2RGB)},
	{{core::raw_format::bayer_gbrg,	core::raw_format::rgb24},		static_cast<int>(CV_BayerGR2RGB)},

	{{core::raw_format::bayer_rggb,	core::raw_format::bgr24},		static_cast<int>(CV_BayerBG2BGR)},
	{{core::raw_format::bayer_bggr,	core::raw_format::bgr24},		static_cast<int>(CV_BayerRG2BGR)},
	{{core::raw_format::bayer_grbg,	core::raw_format::bgr24},		static_cast<int>(CV_BayerGB2BGR)},
	{{core::raw_format::bayer_gbrg,	core::raw_format::bgr24},		static_cast<int>(CV_BayerGR2BGR)},

	{{core::raw_format::bayer_rggb,	core::raw_format::rgb48},		static_cast<int>(CV_BayerBG2RGB)},
	{{core::raw_format::bayer_bggr,	core::raw_format::rgb48},		static_cast<int>(CV_BayerRG2RGB)},
	{{core::raw_format::bayer_grbg,	core::raw_format::rgb48},		static_cast<int>(CV_BayerGB2RGB)},
	{{core::raw_format::bayer_gbrg,	core::raw_format::rgb48},		static_cast<int>(CV_BayerGR2RGB)},

	{{core::raw_format::bayer_rggb,	core::raw_format::bgr48},		static_cast<int>(CV_BayerBG2BGR)},
	{{core::raw_format::bayer_bggr,	core::raw_format::bgr48},		static_cast<int>(CV_BayerRG2BGR)},
	{{core::raw_format::bayer_grbg,	core::raw_format::bgr48},		static_cast<int>(CV_BayerGB2BGR)},
	{{core::raw_format::bayer_gbrg,	core::raw_format::bgr48},		static_cast<int>(CV_BayerGR2BGR)},
};
}

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* OpenCV_H_ */
