/*!
 * @file 		MjpegToJpeg.cpp
 * @author 		<Your name>
 * @date		01.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "MjpegToJpeg.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"
#include <cassert>
#include <iostream>
namespace yuri {
namespace mjpeg_to_jpeg {


IOTHREAD_GENERATOR(MjpegToJpeg)

MODULE_REGISTRATION_BEGIN("mjpeg_to_jpeg")
		REGISTER_IOTHREAD("mjpeg_to_jpeg",MjpegToJpeg)
		REGISTER_CONVERTER(core::compressed_frame::mjpg, core::compressed_frame::jpeg, "mjpeg_to_jpeg", 10);
MODULE_REGISTRATION_END()

core::Parameters MjpegToJpeg::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::CompressedVideoFrame>::configure();
	p.set_description("MJPEG to JPEG convertor. This node simply adds HUffman tables to mjpeg frames missing one");
	return p;
}

namespace {

const uint8_t huffman_table_marker = 0xc4;
const uint8_t sos_marker = 0xda;

// Default Huffman tables (according to http://www.daevius.com/information-jpeg-file-format)

const uint8_t huffman_tables[] = {
/* Luminance DC */
// Table 0, DC
	0x00,

// bits
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,

//values
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,

/* Luminance AC */
// Table 0, AC
    0x10,
//bits

    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d,

// values
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,

/* Chrominance DC */
// Table 1, DC
	0x01,
// bits
    0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,

// values
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,

// Table 1, AC
	0x11,
/* Chrominance AC */
//bits
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77,

//values
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};


template<class Iter>
uint8_t get_header_type(Iter it) {
	if (*it!=0xFF) return 0;
	return static_cast<uint8_t>(*(it+1));
}
template<class Iter>
size_t get_header_size(Iter it) {
	return (*(it+2) << 8) | *(it+3);
}
template<class Iter>
Iter get_hext_header(Iter start, Iter end) {
	const uint8_t header_type = get_header_type(start);
	if (!header_type) return end;
	if (header_type == 0xd8 || header_type == 0xd9) return start+2; // Start/end of image
	size_t offset = get_header_size(start);
	if (std::distance(start, end) < offset) return end;
	return start+offset+2;
}

template<class Iter>
bool find_huffman_table(Iter start, Iter end)
{
	while (start != end) {
		const uint8_t marker_type = get_header_type(start);
		if (marker_type == huffman_table_marker) return true;
		if (marker_type == sos_marker) return false;
		start = get_hext_header(start, end);
	}
	return false;
}

}


MjpegToJpeg::MjpegToJpeg(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::CompressedVideoFrame>(log_,parent,std::string("mjpeg_to_jpeg"))
{
	IOTHREAD_INIT(parameters)
}

MjpegToJpeg::~MjpegToJpeg() noexcept
{
}

core::pFrame MjpegToJpeg::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	if (target_format != core::compressed_frame::jpeg) return {};
	core::pCompressedVideoFrame f = dynamic_pointer_cast<core::CompressedVideoFrame>(input_frame);
	if (!f) return {};
	return do_special_single_step(f);
}
core::pFrame MjpegToJpeg::do_special_single_step(const core::pCompressedVideoFrame& frame)
{
	if (!frame || frame->get_format() != core::compressed_frame::mjpg) return {};

	// If the frame already has a huffman table, then there's no work to do, we just have to change it's identifier to mjpeg
	if (find_huffman_table(frame->begin(), frame->end()))
		return core::CompressedVideoFrame::create_empty(
			core::compressed_frame::jpeg, frame->get_resolution(), frame->data(), frame->size());


	/*
	 * The code bellow tries to find SOS marker and insert default huffman table right before it.
	 */
	auto it = frame->begin();
	while (it!=frame->end()) {
		if (get_header_type(it) == sos_marker) break;
		it = get_hext_header(it, frame->end());
	}
	if (it == frame->end()) return {};
	// Create an empty frame for the new jpeg frame
	core::pCompressedVideoFrame frame_out = core::CompressedVideoFrame::create_empty(
			core::compressed_frame::jpeg, frame->get_resolution(), frame->size()+4+sizeof(huffman_tables));

	// Copy everything before SOS marker
	auto it2 = std::copy(frame->begin(),it,frame_out->begin());
	// Add DHT marker
	*it2++ = 0xff;
	*it2++ = huffman_table_marker;
	const size_t hheader_size = sizeof(huffman_tables)+2;
	*it2++ = static_cast<uint8_t>((hheader_size>>8)&0xFF);
	*it2++ = static_cast<uint8_t>(hheader_size&0xFF);
	// Copy Huffman tables
	it2 = std::copy(huffman_tables, huffman_tables+sizeof(huffman_tables), it2);
	// Copy SOS marker and the rest of the frame
	it2 = std::copy(it,frame->end(), it2);
	return frame_out;
}


bool MjpegToJpeg::set_param(const core::Parameter& param)
{
	return core::SpecializedIOFilter<core::CompressedVideoFrame>::set_param(param);
}

} /* namespace mjpeg_to_jpeg */
} /* namespace yuri */
