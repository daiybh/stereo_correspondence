/*
 * pipe_types.h
 *
 *  Created on: 15.9.2013
 *      Author: neneko
 */

#ifndef PIPE_TYPES_H_
#define PIPE_TYPES_H_
#include "yuri/core/utils/new_types.h"
namespace yuri {
namespace core {
/*namespace pixel_format {
const pix_fmt_t r 			= 0;
const pix_fmt_t g 			= 0;
const pix_fmt_t b 			= 0;
const pix_fmt_t y 			= 0;
const pix_fmt_t u 			= 0;
const pix_fmt_t v 			= 0;

}*/

namespace raw_format {

const format_t unknown		= 0;
// SIngle component formats
const format_t r8			= 0x1;
const format_t r16			= 0x2;
const format_t g8			= 0x3;
const format_t g16			= 0x4;
const format_t b8			= 0x5;
const format_t b16			= 0x6;
const format_t y8			= 0x7;
const format_t y16			= 0x8;
const format_t u8			= 0x9;
const format_t u16			= 0xa;
const format_t v8			= 0xb;
const format_t v16			= 0xc;
const format_t depth8		= 0xd;
const format_t depth16		= 0xe;
// Packed formats (1 plane)
const format_t rgb8			= 0x100;	// RGB 3:3:2
const format_t rgb15		= 0x101;	// RGB 5:5:5
const format_t rgb16		= 0x102;	// RGB 5:6:5
const format_t rgb24		= 0x103;	// RGB 8:8:8
const format_t rgb48		= 0x104;	// RGB 16:16:16
const format_t bgr8			= 0x105;	// BGR 2:3:3
const format_t bgr15		= 0x106;	// BGR 5:5:5
const format_t bgr16		= 0x107;	// BGR 5:6:5
const format_t bgr24		= 0x108;	// BGR 8:8:8
const format_t bgr48		= 0x109;	// BGR 16:16:16

const format_t rgba16		= 0x10a;	// RGBA 5:5:5:1
const format_t rgba32		= 0x10b;	// RGBA 8:8:8:8
const format_t rgba64		= 0x10c;	// RGBA 16:16:16:16
const format_t abgr16		= 0x10d;	// ABGR 5:5:5:1
const format_t abgr32		= 0x10e;	// ABGR 8:8:8:8
const format_t abgr64		= 0x10f;	// ABGR 16:16:16:16

const format_t rgb_r10k		= 0x110;	// RGBx 10:10:10:2
const format_t bgr_r10k		= 0x111;	// BGRx 10:10:10:2


const format_t yuv411		= 0x200;	// YYUYYV	4pixels, 8bit
const format_t yvu411		= 0x201;	// YYVYYU	4pixels, 8bit
const format_t yuyv422		= 0x202;	// YUYV		2pixels, 8bit
const format_t yvyu422		= 0x203;	// YVYU		2pixels, 8bit
const format_t uyvy422		= 0x204;	// UYVY		2pixels, 8bit
const format_t vyuy422		= 0x205;	// VYUY		2pixels, 8bit
const format_t yuv444		= 0x206;	// YUV		1 pixel, 8bit

const format_t yuv422_v210	= 0x220;	// YUVx	10:10:10:2
const format_t yvu422_v210	= 0x221;	// YVUx	10:10:10:2

const format_t xyz			= 0x300;	// XYZ 8:8:8
const format_t bayer_rggb	= 0x301;	// BAYER pattern RGGB
const format_t bayer_bggr	= 0x302;	// BAYER pattern BGGR
const format_t bayer_grbg	= 0x303;	// BAYER pattern GRBG
const format_t bayer_gbrg	= 0x304;	// BAYER pattern GBRG

// Planar formats
const format_t rgb24p		= 0x400;	// RGB 8:8:8 (planar)
const format_t rgb48p		= 0x401;	// RGB 16:16:16 (planar)
const format_t bgr24p		= 0x402;	// BGR 8:8:8 (planar)
const format_t bgr48p		= 0x403;	// BGR 16:16:16 (planar)
const format_t rgba32p		= 0x404;	// RGBA 8:8:8:8 (planar)
const format_t rgba64p		= 0x405;	// RGBA 16:16:16:`6 (planar)
const format_t abgr32p		= 0x406;	// ABGR 8:8:8:8 (planar)
const format_t abgr64p		= 0x407;	// ABGR 16:16:16:16 (planar)


const format_t yuv444p		= 0x400;	// RGB 8:8:8 (planar)


const format_t user_start	= 0x1000;

}

namespace image_format {

}
}
}


#endif /* PIPE_TYPES_H_ */