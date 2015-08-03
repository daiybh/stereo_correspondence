/*!
 * @file 		Split.cpp
 * @author 		Zdenek Travnicek
 * @date 		30.3.2011
 * @date		16.2.2013
 * @date		14.8.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2011 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 */


#include "Split.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils.h"
#include "yuri/core/frame/raw_frame_params.h"
namespace yuri {

namespace split {

MODULE_REGISTRATION_BEGIN("split")
		REGISTER_IOTHREAD("split",Split)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(Split)

core::Parameters Split::configure()
{
	auto p = base_type::configure();
	p["x"]["number of splits in X axis"]=2;
	p["y"]["number of splits in Y axis"]=1;
	return p;
}


Split::Split(log::Log &_log, core::pwThreadBase parent,core::Parameters parameters):
			base_type(_log,parent,2,"split"),x_(2),y_(1)
{
	IOTHREAD_INIT(parameters);
	resize(1,x_+y_);
}

Split::~Split() noexcept {
}

std::vector<core::pFrame> Split::do_special_step(std::tuple<core::pRawVideoFrame> frames)
{
	const auto& frame 	= std::get<0>(frames);
	const yuri::size_t height		= frame->get_height();
	const yuri::size_t width 		= frame->get_width();
	const format_t format 			= frame->get_format();
	const auto& fi					= core::raw_format::get_format_info(format);

	if (fi.planes.size() != 1) {
		log[log::warning] << "Input frames has to have only single image plane";
		return {};
	}

	const auto bpp = core::raw_format::get_fmt_bpp(format, 0);
	if (bpp % 8) {
		log[log::warning] << "Input frames has to have bit depth divisible by 8";
		return {};
	}
	yuri::size_t Bpp = bpp >> 3;
	std::vector<core::pFrame> output;
	const size_t line_size_in		= width * Bpp;

	size_t y_pos = 0;
	const auto data_begin = PLANE_DATA(frame,0).begin(); ///< Iterator to the beginning of the input data
	for (size_t sy = 0; sy < y_; ++sy) { // Iterate over all rows of output
		const size_t split_h = (height - y_pos) / (y_ - sy); ///< Height of output frames in current row.
		const auto data_row_begin = data_begin + line_size_in * y_pos; ///< Iterator to the beginning of the input data for first line of this output row
		size_t x_pos = 0;
		for (size_t sx = 0; sx < x_; ++sx) { // Iterate over all columns of output
			const size_t split_w 	= (width - x_pos) / (x_ - sx); ///< Width of output frames in current row.
			auto data_col_begin 	= data_row_begin + x_pos * Bpp; ///< Iterator to the beginning of the input data for this output frame
			const size_t line_size	= split_w * Bpp;
			x_pos += split_w;
			auto out = core::RawVideoFrame::create_empty(format, {split_w, split_h});
			auto output_data = PLANE_DATA(out,0).begin();
			for (size_t line = 0; line < split_h; ++line) {
				std::copy(data_col_begin, data_col_begin + split_w*Bpp, output_data);
				output_data += line_size;
				data_col_begin += line_size_in;
			}
			output.push_back(std::move(out));
		}
		y_pos += split_h;
	}
	return output;
}

bool Split::set_param(const core::Parameter &parameter)
{
	if (assign_parameters(parameter)
			(x_, "x")
			(y_, "y"))
			return true;
	return base_type::set_param(parameter);
}
}

}
