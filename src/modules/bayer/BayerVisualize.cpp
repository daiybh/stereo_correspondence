/*!
 * @file 		BayerVisualize.cpp
 * @author 		<Your name>
 * @date		31.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "BayerVisualize.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri {
namespace bayer {


IOTHREAD_GENERATOR(BayerVisualize)

MODULE_REGISTRATION_BEGIN("bayer")
		REGISTER_IOTHREAD("bayer_visualize",BayerVisualize)
MODULE_REGISTRATION_END()

core::Parameters BayerVisualize::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("BayerVisualize");
	return p;
}


BayerVisualize::BayerVisualize(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("bayer_visualize"))
{
	IOTHREAD_INIT(parameters)
}

BayerVisualize::~BayerVisualize() noexcept
{
}

namespace {

struct R {
	static void process(const uint8_t*& in, uint8_t*& out) {
		*out++=*in++;
		*out++=0;
		*out++=0;
	}
};
struct G {
	static void process(const uint8_t*& in, uint8_t*& out) {
		*out++=0;
		*out++=*in++;
		*out++=0;
	}
};
struct B {
	static void process(const uint8_t*& in, uint8_t*& out) {
		*out++=0;
		*out++=0;
		*out++=*in++;
	}
};

template<class A0, class A1, class B0, class B1>
void visualize(const uint8_t* in, uint8_t* out, resolution_t res)
{
	const dimension_t lines = res.height&~0x1;
	const dimension_t columns = res.width&~0x1;
	for (dimension_t line =0; line < lines; line +=2) {
		for (dimension_t col = 0; col < columns; col+=2) {
			A0::process(in, out);
			A1::process(in, out);
		}
		for (dimension_t col = 0; col < columns; col+=2) {
			B0::process(in, out);
			B1::process(in, out);
		}
	}
}


}

core::pFrame BayerVisualize::do_special_single_step(const core::pRawVideoFrame& frame)
{
	using namespace core::raw_format;
	const auto res = frame->get_resolution();
	auto outframe = core::RawVideoFrame::create_empty(rgb24, res);
	switch (frame->get_format()) {
		case bayer_bggr:
			visualize<B,G,G,R>(PLANE_RAW_DATA(frame,0), PLANE_RAW_DATA(outframe,0), res);
			break;
		case bayer_rggb:
			visualize<R,G,G,B>(PLANE_RAW_DATA(frame,0), PLANE_RAW_DATA(outframe,0), res);
			break;
		case bayer_gbrg:
			visualize<G,B,R,G>(PLANE_RAW_DATA(frame,0), PLANE_RAW_DATA(outframe,0), res);
			break;
		case bayer_grbg:
			visualize<G,R,B,G>(PLANE_RAW_DATA(frame,0), PLANE_RAW_DATA(outframe,0), res);
			break;
		default:
			return {};
	}

	return outframe;
}
bool BayerVisualize::set_param(const core::Parameter& param)
{
	return base_type::set_param(param);
}

} /* namespace bayer */
} /* namespace yuri */
