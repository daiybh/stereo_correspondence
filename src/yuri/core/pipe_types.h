/*!
 * @file 		pipe_types.h
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef PIPE_TYPES_H_
#define PIPE_TYPES_H_
#include <string>
#include <vector>
#include "yuri/core/types.h"

namespace yuri {
typedef shared_ptr<struct FormatInfo> FormatInfo_t;

typedef struct EXPORT FormatInfo {
	static FormatInfo_t raw_format (std::string name, std::vector<std::string> shortnames,
			bool compressed=false, size_t bpp=0,
			std::vector<yuri::size_t> component_depths=std::vector<yuri::size_t>(),
			std::vector<std::string> components=std::vector<std::string>(),
			std::vector<yuri::size_t> xsub = std::vector<yuri::size_t>(1,1),
			std::vector<yuri::size_t> ysub = std::vector<yuri::size_t>(1,1));
	static FormatInfo_t image_format (std::string name, std::vector<std::string> shortnames,
				std::vector<std::string> mime_types);
	static FormatInfo_t video_format (std::string name, std::vector<std::string> shortnames,
				std::vector<std::string> mime_types);
protected:
	FormatInfo(yuri::format_t type,std::string name, std::vector<std::string> shortnames,
			bool compressed=false, size_t bpp=0,
			std::vector<yuri::size_t> component_depths=std::vector<yuri::size_t>(),
			std::vector<std::string> components=std::vector<std::string>(),
			std::vector<yuri::size_t> xsub = std::vector<yuri::size_t>(1,1),
			std::vector<yuri::size_t> ysub = std::vector<yuri::size_t>(1,1)
			)
	:type(type),long_name(name),short_names(shortnames),bpp(bpp),
	 components(components), compressed(compressed),
	 component_depths(component_depths), plane_x_subs(xsub), plane_y_subs(ysub)
	{
		planes = plane_x_subs.size();
		if (!compressed) {
			if (!component_depths.size()) component_depths=std::vector<yuri::size_t>(planes,bpp/planes);
		}
		assert(planes == plane_y_subs.size());
	}
public:
	/// \brief Type of the format
	yuri::format_t type;
	/// \brief Name of the format
std::string long_name;
	/// \brief std::vector of short names that could be used to represent the format
	std::vector<std::string> short_names;
	/// \brief Bits per pixel (summed from all planes)
	yuri::size_t bpp;
	/// \brief Number of image planes
	yuri::size_t planes;

    /// \brief Components as they appear in the bytestream
	/**
	 * R - Red channel
	 * G - Green channel
	 * B - Blue channel
	 * A - Alpha channel
	 * Y - Luminance
	 * U - Chroma Y-R
	 * V - Chroma Y-B
	 * D - Depth
	 *
	 * If there's more than 1 plane, each component represents one plane
	 * If there's exactly 1 plane the components specify one or more bytes, as found in the byte stream.
	 * Does not mean anything when using compressed images
	 */
	std::vector<std::string> components;
	/// \brief Is bytestream compressed or custom
	bool compressed;
	/// \brief bitdepths of individual components
	std::vector<yuri::size_t> component_depths;
	/// \brief Subsampling in X direction
	std::vector<yuri::size_t> plane_x_subs;
	/// \brief Subsampling in Y direction
	std::vector<yuri::size_t> plane_y_subs;
	/// \brief Mime types for this format
	std::vector<std::string> mime_types;
} FormatInfo;

}
/*
 *		Pipe drop policies
 */

#define YURI_DROP_NONE									0
#define YURI_DROP_COUNT									1
#define YURI_DROP_SIZE									2
/*
 *		Data types
 */

#define YURI_TYPE_NONE									-1
#define YURI_TYPE_VIDEO									0
#define YURI_TYPE_IMAGE									YURI_TYPE_VIDEO
#define YURI_TYPE_AUDIO									1
#define YURI_TYPE_TEXT									2

/*
 * 		Data formats
 */

#define YURI_FMT_NONE -1

#define YURI_FMT										0
#define YURI_FMT_RGB 									YURI_FMT + 1
#define YURI_FMT_RGB24 									YURI_FMT_RGB
#define YURI_FMT_RGB_PLANAR 							YURI_FMT + 2
#define YURI_FMT_RGB24_PLANAR 							YURI_FMT_RGB_PLANAR
#define YURI_FMT_RGBA 									YURI_FMT + 3
#define YURI_FMT_RGB32 									YURI_FMT_RGBA
#define YURI_FMT_RGBA_PLANAR 							YURI_FMT + 4
#define YURI_FMT_RGB32_PLANAR 							YURI_FMT_RGBA_PLANAR
#define YURI_FMT_BGR									YURI_FMT + 5
#define YURI_FMT_BGRA									YURI_FMT + 6
#define YURI_FMT_DEPTH8									YURI_FMT + 7
#define YURI_FMT_DEPTH									YURI_FMT_DEPTH8
#define YURI_FMT_DEPTH16								YURI_FMT + 8
#define YURI_FMT_SINGLE_COMPONENT						YURI_FMT + 9
#define YURI_FMT_SINGLE_COMPONENT16						YURI_FMT + 10
#define YURI_FMT_DXT1									YURI_FMT + 11
#define YURI_FMT_DXT2									YURI_FMT + 12
#define YURI_FMT_DXT3									YURI_FMT + 13
#define YURI_FMT_DXT4									YURI_FMT + 14
#define YURI_FMT_DXT5									YURI_FMT + 15
#define YURI_FMT_RED8									YURI_FMT + 16
#define YURI_FMT_GREEN8									YURI_FMT + 17
#define YURI_FMT_BLUE8									YURI_FMT + 18
#define YURI_FMT_RED16									YURI_FMT + 19
#define YURI_FMT_GREEN16								YURI_FMT + 20
#define YURI_FMT_BLUE16									YURI_FMT + 21
#define YURI_FMT_Y8										YURI_FMT + 22
#define YURI_FMT_U8										YURI_FMT + 23
#define YURI_FMT_V8										YURI_FMT + 24
#define YURI_FMT_Y16									YURI_FMT + 25
#define YURI_FMT_U16									YURI_FMT + 26
#define YURI_FMT_V16									YURI_FMT + 27
#define YURI_FMT_YUV411									YURI_FMT + 28
#define YURI_FMT_YUV422									YURI_FMT + 29
#define YURI_FMT_YUV444									YURI_FMT + 30
#define YURI_FMT_YUV422_PLANAR							YURI_FMT + 31
#define YURI_FMT_YUV420_PLANAR							YURI_FMT + 32
#define YURI_FMT_YUV444_PLANAR							YURI_FMT + 33
#define YURI_FMT_V210									YURI_FMT + 34
#define YURI_FMT_V210_MVTP								YURI_FMT + 35
#define YURI_FMT_DXT1_WITH_MIPMAPS						YURI_FMT + 36
#define YURI_FMT_DXT2_WITH_MIPMAPS						YURI_FMT + 37
#define YURI_FMT_DXT3_WITH_MIPMAPS						YURI_FMT + 38
#define YURI_FMT_DXT4_WITH_MIPMAPS						YURI_FMT + 39
#define YURI_FMT_DXT5_WITH_MIPMAPS						YURI_FMT + 40
#define YURI_FMT_MVTP_FULL_FRAME						YURI_FMT + 41
#define YURI_FMT_MVTP_AUX_DATA							YURI_FMT + 42
#define YURI_FMT_R210									YURI_FMT + 43
#define YURI_FMT_BAYER_RGGB								YURI_FMT + 44
#define YURI_FMT_BAYER_BGGR								YURI_FMT + 45
#define YURI_FMT_BAYER_GRBG								YURI_FMT + 46
#define YURI_FMT_BAYER_GBRG								YURI_FMT + 46

#define YURI_FMT_MAX									YURI_FMT_BAYER_GBRG

#define YURI_IMAGE										0x1000
#define YURI_IMAGE_JPEG									YURI_IMAGE + 1
#define YURI_IMAGE_PNG									YURI_IMAGE + 2
#define YURI_IMAGE_GIF									YURI_IMAGE + 3
#define	YURI_IMAGE_TIFF									YURI_IMAGE + 4
#define YURI_IMAGE_DDS									YURI_IMAGE + 5
#define YURI_IMAGE_JPEG2K								YURI_IMAGE + 6
#define YURI_IMAGE_MAX									YURI_IMAGE_JPEG2K

#define	YURI_VIDEO										0x2000
#define YURI_VIDEO_MPEG2								YURI_VIDEO + 1
#define YURI_VIDEO_MPEG1								YURI_VIDEO + 2
#define YURI_VIDEO_HUFFYUV								YURI_VIDEO + 3
#define YURI_VIDEO_DV									YURI_VIDEO + 4
#define YURI_VIDEO_MJPEG								YURI_VIDEO + 5
#define YURI_VIDEO_MPEGTS								YURI_VIDEO + 6
#define YURI_VIDEO_H264									YURI_VIDEO + 7
#define YURI_VIDEO_FLASH								YURI_VIDEO + 8
#define YURI_VIDEO_DIRAC								YURI_VIDEO + 9
#define YURI_VIDEO_H263									YURI_VIDEO + 10
#define YURI_VIDEO_H263PLUS								YURI_VIDEO + 11
#define YURI_VIDEO_THEORA								YURI_VIDEO + 12
#define YURI_VIDEO_VP8									YURI_VIDEO + 13
#define YURI_VIDEO_FFV1									YURI_VIDEO + 14
#define YURI_VIDEO_MAX									YURI_VIDEO_FFV1

#define YURI_AUDIO										0x3000
#define YURI_AUDIO_PCM_U8								YURI_AUDIO
#define YURI_AUDIO_PCM_S16_LE							YURI_AUDIO + 1
#define YURI_AUDIO_PCM_S16_BE							YURI_AUDIO + 2
#define YURI_AUDIO_PCM_S24_LE							YURI_AUDIO + 3
#define YURI_AUDIO_PCM_S24_BE							YURI_AUDIO + 4
#define YURI_AUDIO_MAX									YURI_AUDIO_PCM_S24_BE

#endif /* PIPE_TYPES_H_ */

