/*!
 * @file 		Pad.cpp
 * @author 		<Your name>
 * @date		14.08.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Pad.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace pad {

REGISTER("pad",Pad)

IO_THREAD_GENERATOR(Pad)

core::pParameters Pad::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Adds letterbox around the image to fill specified dimensions. Image width should be multiply of 4");
	p->set_max_pipes(1,1);
	(*p)["width"]["Width of the destination image"] = 800;
	(*p)["height"]["Height of the destination image"] = 600;
	(*p)["halign"]["Horizontal alignment of the image inside the canvas. (center, left, right)"]=std::string("center");
	(*p)["valign"]["Vertical alignment of the image inside the canvas. (center, top, bottom)"]=std::string("center");
	return p;
}


Pad::Pad(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOFilter(log_,parent,std::string("pad")),
width_(800),height_(600),halign_(horizontal_alignment_t::center),
valign_(vertical_alignment_t::center)
{
	IO_THREAD_INIT("pad")
}

Pad::~Pad()
{
}

namespace {
size_t count_empty_lines_top(size_t height_in, size_t height_out, vertical_alignment_t valign)
{
	if (valign == vertical_alignment_t::top) return 0;
	if (valign == vertical_alignment_t::bottom) return height_out>height_in?height_out-height_in:0;
	return height_out>height_in?(height_out-height_in)/2:0;
}
size_t count_empty_lines_bottom(size_t height_in, size_t height_out, size_t skip_start, vertical_alignment_t valign)
{
	if (valign == vertical_alignment_t::bottom) return 0;
	if (valign == vertical_alignment_t::top) return height_out>height_in?height_out-height_in:0;
	return height_out>height_in?height_out-height_in-skip_start:0;
}
size_t count_empty_cols_left(size_t width_in, size_t width_out, horizontal_alignment_t halign)
{
	if (halign == horizontal_alignment_t::left) return 0;
	if (halign == horizontal_alignment_t::right) return width_out>width_in?width_out-width_in:0;
	return width_out>width_in?(width_out-width_in)/2:0;
}
size_t count_empty_cols_right(size_t width_in, size_t width_out, size_t skip_left, horizontal_alignment_t halign)
{
	if (halign == horizontal_alignment_t::right) return 0;
	if (halign == horizontal_alignment_t::left) return width_out>width_in?width_out-width_in:0;
	return width_out>width_in?width_out-width_in-skip_left:0;
}

template<class Iter, class value_type = typename std::iterator_traits<Iter>::value_type>
void fill_pattern(Iter start, const Iter& end, const std::vector<value_type> pattern)
{
	const size_t pattern_size = pattern.size();
	assert(pattern_size > 0);
	const auto pat_start = pattern.begin();
	const auto pat_end = pattern.end();

	size_t remaining = std::distance(start, end);

	while (remaining > pattern_size) {
		std::copy(pat_start, pat_end, start);
		std::advance(start, pattern_size);
		remaining-=pattern_size;
	}
	std::copy(pat_start, pat_start+remaining, start);
}
/*!
 * Function to fill in black pixels - either 0s or specialized pattern
 * @param start		Iterator to beginning of the range
 * @param end		Iterator to the end of the range
 * @param format	format of the pixels
 */
template<class Iter>
void fill_black(Iter start, const Iter& end, format_t format)
{
	switch (format) {
		case YURI_FMT_YUV444:
			fill_pattern(start, end, {0,128,128});
			break;
		case YURI_FMT_YUV422:
		case YURI_FMT_YVYU422:
			fill_pattern(start, end, {0,128});
			break;
		case YURI_FMT_UYVY422:
		case YURI_FMT_VYUY422:
			fill_pattern(start, end, {128, 0});
			break;
		case YURI_FMT_RGB:
		case YURI_FMT_RGBA:
		case YURI_FMT_BGR:
		case YURI_FMT_BGRA:
		default:std::fill(start,end,0);break;

	}
}
}


core::pBasicFrame Pad::do_simple_single_step(const core::pBasicFrame& frame)
{
	const yuri::size_t height		= frame->get_height();
	const yuri::size_t width 		= frame->get_width();
	const format_t format 			= frame->get_format();
	const FormatInfo_t info 		= core::BasicPipe::get_format_info(format);

	if (info->bpp % 8) {
		log[log::warning] << "Input frames has to have bit depth divisible by 8";
		return {};
	}
	if (info->planes != 1) {
		log[log::warning] << "Input frames has to have only single image plane";
		return {};
	}
	const yuri::size_t Bpp 			= info->bpp >> 3;

	const size_t line_size_in		= width  * Bpp;
	const size_t line_size_out		= width_ * Bpp;

	const size_t blank_lines_top 	= count_empty_lines_top(height, height_, valign_);
	const size_t skip_lines_top 	= count_empty_lines_top(height_, height, valign_);
	const size_t blank_lines_bottom = count_empty_lines_bottom(height, height_, blank_lines_top, valign_);
	//const size_t skip_lines_bottom 	= count_empty_lines_bottom(height_, height, skip_lines_top, valign_);

	const size_t blank_cols_left 	= count_empty_cols_left(width, width_, halign_);
	const size_t skip_cols_left 	= count_empty_cols_left(width_, width, halign_);
	const size_t blank_cols_right 	= count_empty_cols_right(width, width_, blank_cols_left, halign_);
	//const size_t skip_cols_right 	= count_empty_cols_right(width_, width, skip_cols_left, halign_);

	const size_t copy_width			= width_ - blank_cols_left - blank_cols_right;
	const size_t copy_size			= copy_width * Bpp;

	log[log::verbose_debug] << "Padding with " << blank_cols_left << " pixels left, " << blank_cols_right << "pixels right, "
				<< blank_lines_top << " pixels on the top and " << blank_lines_bottom << " pixels on the bottom";

	core::pBasicFrame output = allocate_empty_frame(format, width_, height_);

	const auto data_in_start		= PLANE_DATA(frame,0).begin()+skip_lines_top*line_size_in+skip_cols_left*Bpp;
	const auto data_in_end			= PLANE_DATA(frame,0).end();
	const auto data_out_start		= PLANE_DATA(output,0).begin();
	// Fill in empty lines at the top
	for (size_t line = 0; line < blank_lines_top; ++line) {
		fill_black(data_out_start+line*line_size_out, data_out_start+(line+1)*line_size_out, format);
	}
	// Fill in empty lines at the bottom
	for (size_t line = height_ - blank_lines_bottom; line < height_; ++line) {
		fill_black(data_out_start+line*line_size_out, data_out_start+(line+1)*line_size_out, format);
	}
	auto data_in = data_in_start;
	// Process all non-empty lines
	for (size_t line = blank_lines_top; line < height_ - blank_lines_bottom; ++line) {
		const auto out_line_start			= data_out_start + line_size_out * line ;
		const auto next_line_start 			= out_line_start + line_size_out;
		const auto out_line_active_start 	= out_line_start + blank_cols_left * Bpp;
		const auto out_line_active_end 		= out_line_active_start + copy_width;
		// Fill in blank pixels at left size
		fill_black(out_line_start, out_line_active_start, format);
		// Fill in blank pixels at left size
		fill_black(out_line_active_end, next_line_start, format);

		// Copy pixels from input
		std::copy(data_in, data_in + copy_size, out_line_active_start);
		std::advance(data_in, line_size_in);


	}

	return output;
}

namespace {
std::map<std::string, horizontal_alignment_t> halign_strings {
	{"left", horizontal_alignment_t::left},
	{"center", horizontal_alignment_t::center},
	{"right", horizontal_alignment_t::right}
};
std::map<std::string, vertical_alignment_t> valign_strings {
	{"top", vertical_alignment_t::top},
	{"center", vertical_alignment_t::center},
	{"bottom", vertical_alignment_t::bottom},
};

}
bool Pad::set_param(const core::Parameter& param)
{
	if (iequals(param.name, "width")) {
		width_ = param.get<size_t>();
	} else if (iequals(param.name, "height")) {
		height_ = param.get<size_t>();
	} else if (iequals(param.name, "halign")) {
		auto it = halign_strings.find(param.get<std::string>());
		if (it == halign_strings.end()) halign_ = horizontal_alignment_t::center;
		else halign_=it->second;
	} else if (iequals(param.name, "valign")) {
		auto it = valign_strings.find(param.get<std::string>());
		if (it == valign_strings.end()) valign_ = vertical_alignment_t::center;
		else valign_=it->second;
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace pad */
} /* namespace yuri */
