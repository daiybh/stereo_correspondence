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
namespace yuri {

namespace split {


REGISTER("split",Split)

IO_THREAD_GENERATOR(Split)

core::pParameters Split::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["x"]["number of splits in X axis"]=2;
	(*p)["y"]["number of splits in Y axis"]=1;
	p->set_max_pipes(1,2);
	p->add_input_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGB);
	return p;
}


Split::Split(log::Log &_log, core::pwThreadBase parent,core::Parameters &parameters):
			BasicMultiIOFilter(_log,parent,1,2,"split"),x_(2),y_(1)
{
	IO_THREAD_INIT("Split");
	resize(1,x_+y_);
}

Split::~Split() {
}

std::vector<core::pBasicFrame> Split::do_single_step(const std::vector<core::pBasicFrame>& frames)
{
	assert(frames.size()==1);
	const core::pBasicFrame frame 	= frames[0];
	const yuri::size_t height		= frame->get_height();
	const yuri::size_t width 		= frame->get_width();
	const format_t format 			= frame->get_format();
	const FormatInfo_t info 		= core::BasicPipe::get_format_info(frame->get_format());

	if (info->bpp % 8) {
		log[log::warning] << "Input frames has to have bit depth divisible by 8";
		return {};
	}
	if (info->planes != 1) {
		log[log::warning] << "Input frames has to have only single image plane";
		return {};
	}
	yuri::size_t Bpp = info->bpp >> 3;
	std::vector<core::pBasicFrame> output;
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
			core::pBasicFrame out = allocate_empty_frame(format, split_w, split_h);
			auto output_data = PLANE_DATA(out,0).begin();
			for (size_t line = 0; line < split_h; ++line) {
				std::copy(data_col_begin, data_col_begin + split_w*Bpp, output_data);
				output_data += line_size;
				data_col_begin += line_size_in;
			}
			output.push_back(out);
		}
		y_pos += split_h;
	}
	return output;
}

bool Split::set_param(const core::Parameter &parameter)
{
	if (iequals(parameter.name,"x")) {
		x_ = parameter.get<size_t>();
	} else if (iequals(parameter.name,"y")) {
		y_ = parameter.get<size_t>();
	} else return core::BasicMultiIOFilter::set_param(parameter);
	return true;
}
}

}
